import importlib.util
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class FullDecoderTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('v10_2_sixteen_source_full_decoder'), 'separate LoRA/full entry missing')
        import v10_2_sixteen_source_full_decoder as entry
        return entry

    def test_staged_lora_fresh_model_only_initialization(self):
        a=self.module().training_args({'resume_checkpoint':'old','lr_warmup_epochs':2},
            initialization='A.pth',output='new',gpu=3,split='fixed.json')
        self.assertEqual(a.resume_checkpoint,'')
        self.assertEqual(a.prompt_generator_checkpoint,'A.pth')
        self.assertEqual((a.epochs,a.train_cases_per_epoch,a.validate_every),(40,250,10))
        self.assertEqual(a.sam_encoder_unfreeze_epoch,6)
        self.assertEqual(a.sam_tuning_mode,'encoder_lora_full_decoder')
        self.assertEqual((a.sam_encoder_lora_rank,a.sam_encoder_lora_alpha),(8,16))
        self.assertEqual((a.encoder_lr,a.decoder_lr),(5e-7,2e-7))

    def test_smoke_forces_real_lora_update(self):
        a=self.module().training_args({'lr_warmup_epochs':2},initialization='A',output='smoke',gpu=3,split='s',smoke=True)
        self.assertEqual(a.sam_encoder_unfreeze_epoch,1)
        self.assertEqual((a.epochs,a.train_cases_per_epoch),(1,16))
        self.assertTrue(a.v10_2_poscap_skip_validation)

    def test_frozen_entry_defaults_remain_frozen(self):
        import v10_2_sixteen_source_train as frozen
        a=frozen.training_args({},initialization='A',output='n',gpu=2,split='s')
        self.assertEqual(a.sam_encoder_unfreeze_epoch,41)

    def test_reject_full_encoder_thaw_and_frozen_decoder(self):
        module=self.module()
        class Parameter:
            def __init__(self,enabled): self.requires_grad=enabled
        class Named:
            def __init__(self,rows): self.rows=rows
            def named_parameters(self): return iter(self.rows)
        sam=Named([('image_encoder.blocks.0.qkv.lora_A',Parameter(True)),
                   ('image_encoder.blocks.0.qkv.lora_B',Parameter(True)),
                   ('image_encoder.blocks.0.qkv.base.weight',Parameter(False))])
        sam.image_encoder=Named(sam.rows)
        sam.sam_mask_decoder=Named([('core_decoder.head.weight',Parameter(True)),('prompt_3d_adapter.weight',Parameter(False))])
        module.assert_full_decoder_lora_policy(sam)
        sam.rows[-1][1].requires_grad=True
        with self.assertRaisesRegex(RuntimeError,'Encoder'): module.assert_full_decoder_lora_policy(sam)
        sam.rows[-1][1].requires_grad=False
        sam.sam_mask_decoder.rows[0][1].requires_grad=False
        with self.assertRaisesRegex(RuntimeError,'Decoder'): module.assert_full_decoder_lora_policy(sam)


if __name__=='__main__': unittest.main()
