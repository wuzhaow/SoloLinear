import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, TimeXer, \
    FITS, DUET,CycleNet,PaiFilter,TexFilter, PatchMLP, SparseTSF, Model
from thop import profile

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'TimeXer': TimeXer,
            'FITS': FITS,
            'DUET': DUET,
            'PatchMLP': PatchMLP,
            'CycleNet': CycleNet,
            'PaiFilter': PaiFilter,
            'TexFilter': TexFilter,
            'SparseTSF': SparseTSF,
            'Model': Model
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        memory_test = False
        if memory_test:
            #在模型运行前后检查
            print(f"运行前显存: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            B = 1
            input = torch.randn(B,args.seq_len ,args.enc_in ).to(self.device)
            #mark = torch.randn(B,args.seq_len ,4).to(self.device)
            #cyc_index = torch.randint(0,365,(B,)).to(self.device)
            if args.model == 'CycleNet':
                cyc_index = torch.randint(0, 365, (B,)).to(self.device)
                macs, params = profile(self.model, inputs=(input, None, None, None, cyc_index))
            elif args.model in ['TimeMixer','TimeXer, iTransformer'] :
                mark = torch.randn(B, args.seq_len, 4).to(self.device)
                macs, params = profile(self.model, inputs=(input, mark, None, None))
            elif args.model in ['PaiFilter', 'TexFilter']:
                macs, params = profile(self.model, inputs=(input, None, None, None))
            else:
                macs,params = profile(self.model,inputs=(input,None,None,None,None))

            print(args.model)

            print(f"运行后显存: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            print(f"运行后显存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"峰值显存: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
            print(f"峰值显存: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

            print(f"Params: {params}")
            print(f"Params: {params / 1e3:.2f}K")
            print(f"Params: {params / 1e6:.2f}M")

            #if macs >= 1e9:
            print(f"MACs: {macs}")
            print(f"{macs / 1e3:.2f}K MACS")
            print(f"{macs / 1e6:.2f}M MACS")
            print(f"{macs / 1e9:.2f}G MACS")

        # elif macs >= 1e6:

        # else:
        #print(f"{macs} MACS")
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
