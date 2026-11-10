import unittest
from tools.five_source_prompt_type_pipeline_v2 import summarize, numeric_error

class AuditTests(unittest.TestCase):
    def test_nonfinite_merged_numbers_fail_closed(self):
        for value in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(ValueError):numeric_error(value,.5)

    def test_paired_metrics_require_all_variants_and_one_actual_batch(self):
        from tools.five_source_prompt_type_runtime import METRICS,VARIANTS
        task=dict(source_name='magic',case_id='hcc/1',global_class_id=133)
        records=[dict(row=dict(task,**{m:.5+i*.1 for m in METRICS}),variant=v,actual_frame_batch=2) for i,v in enumerate(VARIANTS)]
        s=summarize(records,[task])
        self.assertAlmostEqual(s['magic']['fg_box_bg_point']['paired_delta_vs_baseline']['dice'],.1)
        with self.assertRaises(ValueError):summarize(records[:-1],[task])
        records[2]['actual_frame_batch']=1
        with self.assertRaises(ValueError):summarize(records,[task])

if __name__=='__main__':unittest.main()
