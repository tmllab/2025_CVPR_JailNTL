from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    """ This TesteModel can be used to generate disguised images for the validation phase of JailNTL."""
    def __init__(self, opt, task_name):
        """Initialize the TestModel class.

        Parameters:
            opt -- stores all the experiment flags
        """
        self.isTrain = False
        BaseModel.__init__(self, opt, task_name)
        self.loss_names = []
        # specify the images you want to save
        self.visual_names = ['real', 'fake']
        # specify the models you want to save to the disk
        self.model_names = 'G'  # only generator is needed
        self.netG = networks.define_G(device=opt.device)
        setattr(self, 'netG', self.netG)  # store netG in self

    def set_input(self, input):
        """Set the input data for the model."""
        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass."""
        self.fake = self.netG(self.real)  # G(real)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
