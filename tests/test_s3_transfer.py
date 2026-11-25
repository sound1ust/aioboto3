"""Transfer behavior that can be checked without an S3 server."""

import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

from boto3.s3.transfer import S3TransferConfig, TransferConfig
import pytest

from aioboto3.s3.inject import upload_fileobj


def make_client():
    return SimpleNamespace(
        create_multipart_upload=AsyncMock(return_value={'UploadId': 'upload-id'}),
        upload_part=AsyncMock(return_value={'ETag': 'etag'}),
        complete_multipart_upload=AsyncMock(),
        abort_multipart_upload=AsyncMock(),
    )


@pytest.mark.parametrize('config_class', [S3TransferConfig, TransferConfig])
@pytest.mark.asyncio
async def test_multipart_transfer_config(config_class):
    client = make_client()
    config = config_class(multipart_threshold=4, multipart_chunksize=4, io_chunksize=4)

    await upload_fileobj(client, BytesIO(b'0123456789'), 'bucket', 'key', Config=config)

    assert client.upload_part.await_count == 3
    assert b''.join(call.kwargs['Body'] for call in client.upload_part.await_args_list) == b'0123456789'
    client.complete_multipart_upload.assert_awaited_once()
    client.abort_multipart_upload.assert_not_awaited()


@pytest.mark.parametrize('cancel_during', ['upload', 'complete', 'callback'])
@pytest.mark.asyncio
async def test_multipart_cancellation_stops_tasks_before_abort(cancel_during):
    before = asyncio.all_tasks()
    started = asyncio.Event()
    stopped = asyncio.Event()
    client = make_client()

    async def block(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def abort(**kwargs):
        assert stopped.is_set()

    if cancel_during == 'upload':
        client.upload_part.side_effect = block
    elif cancel_during == 'complete':
        client.complete_multipart_upload.side_effect = block
    client.abort_multipart_upload.side_effect = abort
    config = TransferConfig(
        multipart_threshold=4, multipart_chunksize=4, io_chunksize=4,
        max_concurrency=1, max_io_queue=1,
    )
    task = asyncio.create_task(upload_fileobj(
        client, BytesIO(b'0123456789'), 'bucket', 'key', Config=config,
        Callback=block if cancel_during == 'callback' else None,
    ))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    client.abort_multipart_upload.assert_awaited_once_with(Bucket='bucket', Key='key', UploadId='upload-id')
    assert asyncio.all_tasks() == before


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_abort():
    before = asyncio.all_tasks()
    uploading = asyncio.Event()
    aborting = asyncio.Event()
    release_abort = asyncio.Event()
    client = make_client()

    async def upload(**kwargs):
        uploading.set()
        await asyncio.Event().wait()

    async def abort(**kwargs):
        aborting.set()
        await release_abort.wait()

    client.upload_part.side_effect = upload
    client.abort_multipart_upload.side_effect = abort
    config = TransferConfig(multipart_threshold=4, multipart_chunksize=4, io_chunksize=4)
    task = asyncio.create_task(upload_fileobj(client, BytesIO(b'0123456789'), 'bucket', 'key', Config=config))
    try:
        await asyncio.wait_for(uploading.wait(), timeout=5)
        task.cancel()
        await asyncio.wait_for(aborting.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release_abort.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
    finally:
        release_abort.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    client.abort_multipart_upload.assert_awaited_once()
    assert asyncio.all_tasks() == before


@pytest.mark.parametrize('failure', ['upload', 'complete'])
@pytest.mark.asyncio
async def test_abort_error_preserves_transfer_error(failure):
    client = make_client()
    error = ValueError('transfer failed')
    operation = client.upload_part if failure == 'upload' else client.complete_multipart_upload
    operation.side_effect = error
    client.abort_multipart_upload.side_effect = RuntimeError('abort failed')
    config = TransferConfig(
        multipart_threshold=4, multipart_chunksize=4, io_chunksize=4,
        max_concurrency=1, max_io_queue=1,
    )

    with pytest.raises(ValueError) as caught:
        await asyncio.wait_for(
            upload_fileobj(client, BytesIO(b'0123456789abcdef'), 'bucket', 'key', Config=config),
            timeout=5,
        )

    assert caught.value is error
    client.abort_multipart_upload.assert_awaited_once()
