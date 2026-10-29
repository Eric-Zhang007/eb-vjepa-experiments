import unittest
import torch
from elaf.v0_prefetch import OrderedBatchPrefetch


class Dataset:
    def get_item(self,index,*,augmentation_generator=None):
        if index<0:raise ValueError('invalid sample')
        # Two camera seeds, same draws as CachedV0Dataset.
        return torch.stack([torch.randint(0,2147483647,(2,),generator=augmentation_generator)
                            for _ in range(2)])+index


class PrefetchTests(unittest.TestCase):
    def test_exact_draws_global_rng_isolation_and_resume(self):
        torch.manual_seed(239);initial=torch.get_rng_state().clone();dataset=Dataset()
        indices=[[i,i+1] for i in range(10)]
        expected=[torch.stack([dataset.get_item(i) for i in batch]) for batch in indices]
        final=torch.get_rng_state().clone();torch.set_rng_state(initial)
        with OrderedBatchPrefetch(dataset,indices,torch.stack,pin_memory=False) as loader:
            seen=[]
            for _ in range(4):
                item=next(loader);seen.append(item.batch)
            self.assertTrue(torch.equal(torch.get_rng_state(),initial))
            saved=item.rng_state.clone()
        # Ahead-of-consumer queue is discarded, and must not affect resume.
        with OrderedBatchPrefetch(dataset,indices[4:],torch.stack,rng_state=saved,pin_memory=False) as loader:
            for item in loader:seen.append(item.batch);torch.set_rng_state(item.rng_state)
        for actual,wanted in zip(seen,expected):self.assertTrue(torch.equal(actual,wanted))
        self.assertTrue(torch.equal(torch.get_rng_state(),final))

    def test_exception_and_full_queue_shutdown(self):
        with OrderedBatchPrefetch(Dataset(),[[-1]],torch.stack,pin_memory=False) as loader:
            with self.assertRaisesRegex(RuntimeError,'invalid sample'):next(loader)
        loader=OrderedBatchPrefetch(Dataset(),([i] for i in range(10000)),torch.stack,pin_memory=False)
        loader.close();self.assertFalse(loader.thread.is_alive())


if __name__=='__main__':unittest.main()
