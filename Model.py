# imports
from ISR.models import RDN, RRDN


# define models
# Pre-trained networks: currently 4 models are available: - RDN: psnr-large, psnr-small, noise-cancel - RRDN: gans
class Model:
    def __init__(self):
        print("Staring initialization..")

        # _________________________BASIC_MODELS_________________________
        # RDN models, PSNR driven..
        # large RDN (PSNR)
        self.rdn_lg = RDN(weights='psnr-large')  # arch_params={'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2}
        # self.rdn_lg = RDN(arch_params={'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2})
        # self.rdn_lg.model.load_weights('semestral_work/weights/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5')

        # # small RDN (PSNR)
        self.rdn_sm = RDN(weights='psnr-small')  # arch_params={'C': 3, 'D': 10, 'G': 64, 'G0': 64, 'x': 2}
        # self.rdn_sm = RDN(arch_params={'C': 3, 'D': 10, 'G': 64, 'G0': 64, 'x': 2})
        # self.rdn_sm.model.load_weights('semestral_work/weights/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5')
        #
        # # _________________________GANS_MODEL_________________________
        # # RRDN model, trained with Adversarial and VGG features losses
        # # RRDN GANS
        self.rrdn = RRDN(weights='gans')  # arch_params={'C': 4, 'D': 3, 'G': 32, 'G0': 32, 'x': 4, 'T': 10}
        # self.rrdn = RRDN(arch_params={'C': 4, 'D': 3, 'G': 32, 'G0': 32, 'x': 4, 'T': 10})
        # self.rrdn.model.load_weights('semestral_work/weights/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5')
        #
        # # _________________________ARTEFACT_CANCELLING_GANS_MODEL_________________________
        # # RDN model, trained with Adversarial and VGG features losses
        # # large RDN, noise cancelling
        self.rdn_lg_nc = RDN(weights='noise-cancel')  # arch_params={'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2}
        # self.rdn_lg_nc = RDN(arch_params={'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2})
        # self.rdn_lg_nc.model.load_weights('semestral_work/weights/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')
        print("Done.")

    @classmethod
    def load_model(cls):
        return cls()

    @property
    def RDN_LG(self):
        return self.rdn_lg

    @property
    def RDN_SM(self):
        return self.rdn_sm

    @property
    def RRDN(self):
        return self.rrdn

    @property
    def RDN_LG_NC(self):
        return self.rdn_lg_nc
