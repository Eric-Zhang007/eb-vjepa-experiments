"""Ordered CPU prefetch with an isolated, checkpointable augmentation RNG.

Checklist: exact sequential draws; bounded queue; consumed-state checkpointing;
producer exceptions reach consumer; shutdown never waits on a full queue.
The consumer must not draw from the global CPU torch RNG between batches.
"""
from __future__ import annotations
from dataclasses import dataclass
import queue
import threading
import traceback
import torch


@dataclass
class PreparedBatch:
    batch: dict
    rng_state: torch.Tensor


@dataclass
class _Failure:
    exception: BaseException
    trace: str


_END = object()


def _pin(value,device):
    if torch.is_tensor(value):
        if value.device.type != 'cpu':
            raise ValueError('Prefetch producer may only prepare CPU tensors')
        return value.pin_memory(device=device)
    if isinstance(value, dict):return {k:_pin(v,device) for k,v in value.items()}
    if isinstance(value, list):return [_pin(v,device) for v in value]
    if isinstance(value, tuple):return tuple(_pin(v,device) for v in value)
    return value


class OrderedBatchPrefetch:
    """Yield PreparedBatch in index_batches order; state excludes queued work.

    Construct only after debug assertions / checkpoint RNG restoration. Dataset
    belongs exclusively to the producer until close(). The caller should set
    torch.set_rng_state(item.rng_state) after consuming each item. Resume creates
    a new producer at the saved micro-step with the saved consumed RNG state.
    """
    def __init__(self,dataset,index_batches,collate_fn,*,rng_state=None,depth=2,pin_memory=True,
                 pin_memory_device='cuda:0'):
        if isinstance(depth,bool) or not isinstance(depth,int) or depth<1:
            raise ValueError('prefetch depth must be a positive integer')
        self.dataset=dataset;self.index_batches=iter(index_batches);self.collate_fn=collate_fn
        self.generator=torch.Generator(device='cpu')
        self.generator.set_state((torch.get_rng_state() if rng_state is None else rng_state).clone())
        self.pin_memory=pin_memory;self.pin_memory_device=pin_memory_device;self.queue=queue.Queue(maxsize=depth)
        self.stopping=threading.Event();self.closed=False
        self.thread=threading.Thread(target=self._produce,name='elaf-cpu-prefetch',daemon=True)
        self.thread.start()

    def _put(self,item):
        while not self.stopping.is_set():
            try:self.queue.put(item,timeout=.1);return True
            except queue.Full:pass
        return False

    def _produce(self):
        try:
            for indices in self.index_batches:
                if self.stopping.is_set():break
                samples=[self.dataset.get_item(i,augmentation_generator=self.generator) for i in indices]
                batch=self.collate_fn(samples)
                if self.pin_memory:batch=_pin(batch,self.pin_memory_device)
                if not self._put(PreparedBatch(batch,self.generator.get_state().clone())):return
            self._put(_END)
        except BaseException as error:
            self._put(_Failure(error,traceback.format_exc()))

    def __iter__(self):return self

    def __next__(self):
        if self.closed:raise StopIteration
        while True:
            try:item=self.queue.get(timeout=.1);break
            except queue.Empty:
                if not self.thread.is_alive():
                    self.close();raise RuntimeError('Prefetch producer exited without completion')
        if item is _END:self.close();raise StopIteration
        if isinstance(item,_Failure):
            self.close()
            raise RuntimeError(f'CPU prefetch failed:\n{item.trace}') from item.exception
        return item

    def close(self):
        self.stopping.set();self.closed=True
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('CPU prefetch did not finish its current batch within 30 seconds')

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


class BatchPrefetcher:
    """Training adapter with an explicit consumed micro-step cursor."""
    def __init__(self,dataset,*,seed,rank,world_size,batch_size,start_micro_step,
                 rng_state,depth=2,pin_memory=True,pin_memory_device=None):
        from functools import partial
        from itertools import count
        from elaf.v0_training import sample_indices
        from elaf.v0_dataset import collate_v0
        self.micro_step=start_micro_step
        batches=(sample_indices(dataset.suite_indices,seed=seed,micro_step=step,
            rank=rank,world_size=world_size,batch_size=batch_size) for step in count(start_micro_step))
        import os
        pin_device=pin_memory_device or f'cuda:{int(os.environ.get("LOCAL_RANK",rank))}'
        self.loader=OrderedBatchPrefetch(dataset,batches,
            partial(collate_v0,pad_token_id=dataset.pad_token_id),rng_state=rng_state,
            depth=depth,pin_memory=pin_memory,pin_memory_device=pin_device)

    def next(self,expected_microstep):
        if expected_microstep!=self.micro_step:
            raise ValueError(f'Prefetch cursor mismatch: expected {self.micro_step}, received {expected_microstep}')
        item=next(self.loader);self.micro_step+=1
        return item.batch,item.rng_state

    def close(self):self.loader.close()
