import torch
import itertools
from JailNTL.utils.image_pool import ImagePool
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import numpy as np          

def guidance_grad(ntl, confidence_loss_func, balance_loss_func, logits_real, logits_fake, x_fake, epsilon=1e-5):
    """
    Compute the gradient of guidance loss using finite difference with 100 perturbed samples.
    """    
    # create perturbations in different 100 random directions
    num_pert = 100
    perturbations = [(torch.randn_like(x_fake) * epsilon) for _ in range(num_pert)]
    
    # calculate loss for perturbed data
    x_fake_perts = [x_fake + perturbation for perturbation in perturbations]
    with torch.no_grad():
        logits_fake_perts = [ntl(F.interpolate(x_fake_pert, size=(64,64), mode='bilinear', align_corners=False)) for x_fake_pert in x_fake_perts]
        loss_conf = [confidence_loss_func(logits_real, logits_fake_pert) for logits_fake_pert in logits_fake_perts]
        loss_balance = [balance_loss_func(logits_real, logits_fake_pert) for logits_fake_pert in logits_fake_perts]
    
    # calculate gradients
    loss_diff_conf = torch.stack(loss_conf).mean(dim=0)
    loss_diff_balance = torch.stack(loss_balance).mean(dim=0)
    grad_conf = (loss_diff_conf / epsilon) * torch.ones_like(x_fake)
    grad_balance = (loss_diff_balance / epsilon) * torch.ones_like(x_fake)

    torch.cuda.empty_cache()
            
    return grad_conf, grad_balance

def ntl_confidence_mae():
    """Calculate the confidence loss."""
    def calculate_confidence(logits):
        # entropy based confidence
        probs = F.softmax(logits, dim=1)
        confidence = torch.sum(-probs * torch.log2(probs+1e-12), dim=1)  # Eq.9 in the paper
        return torch.mean(confidence)
    def calculate_confidence_mae(logits_real, logits_fake):
        with torch.no_grad():
            confidence_real = calculate_confidence(logits_real)
            confidence_fake = calculate_confidence(logits_fake)
            mae = torch.nn.functional.l1_loss(confidence_real, confidence_fake) # Eq.10 in the paper
        torch.cuda.empty_cache()
        return mae
    return calculate_confidence_mae

def ntl_class_balance_mae():
    """Calculate the class balance loss."""
    def calculate_class_entropy(logits):
        _, class_counts = torch.unique(logits.max(1)[1], return_counts=True)
        class_prob = class_counts.float() / logits.size(0)  # Eq.11 in the paper
        class_prob = torch.cat([class_prob, torch.zeros(logits.size(1) - class_prob.size(0), device=logits.device)])
        entropy = -(class_prob * torch.log2(class_prob+1e-12)).sum()  # Eq.12 in the paper
        return entropy
    def calculate_class_mae(logits_real, logits_fake):
        with torch.no_grad():
            class_entropy_real = calculate_class_entropy(logits_real)
            class_entropy_fake = calculate_class_entropy(logits_fake)
            mae = torch.nn.functional.l1_loss(class_entropy_real, class_entropy_fake)  # Eq.13 in the paper
        torch.cuda.empty_cache()
        return mae
    return calculate_class_mae

class DisguisingModel(BaseModel):
    """
    This class implements the disguising model for jailbreaking NTL.
    Symbols - Code (vs. paper):
        - network: G_A (f_g), G_B (f_g'), D_A (f_d), D_B (f_d')
        - data: A (D_u), B (D_a)
        - loss: 
            - loss_G_A (GAN Loss), Loss_G_B (GAN Loss, bidirectional)
            - loss_cycle_A (Feedback Loss), loss_cycle_B (Feedback Loss, bidirectional)
            - loss_idt_A (Identity Loss), loss_idt_B (Identity Loss, bidirectional)
            - loss_confidence_B (Confidence Loss, model-guided)
            - loss_balance_B (Class Balance Loss, model-guided)
    """
    def __init__(self, opt, task_name, ntl_model=None):
        """Initialize the DisguisingModel class."""
        self.isTrain = True
        self.lambda_identity = 0.5
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        BaseModel.__init__(self, opt, task_name)
        self.ntl = ntl_model
        
        # specify the training losses you want to print out
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        if self.ntl is not None:
            self.loss_names += ['confidence_B', 'balance_B']
            
        # specify the images you want to save.
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        self.visual_names = visual_names_A + visual_names_B 
        
        # specify the models you want to save to the disk.
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']

        # define networks (both Generators and discriminators)
        # define generators
        self.netG_A = networks.define_G(device=opt.device)
        self.netG_B = networks.define_G(device=opt.device)
        # define discriminators
        self.netD_A = networks.define_D(device=opt.device)
        self.netD_B = networks.define_D(device=opt.device)
        
        # create image buffer to store previously generated images
        self.fake_A_pool = ImagePool()
        self.fake_B_pool = ImagePool()
        
        # define loss functions
        self.criterionGAN = networks.GANLoss().to(self.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.ntl_confidence_mae = ntl_confidence_mae()
        self.ntl_class_mae = ntl_class_balance_mae()
        
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Set the input data for the model."""
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        if self.ntl is not None:
            with torch.no_grad():
                # for stl, cifar, visda - size 64x64, for digits - size 32x32
                self.logits_fake_B = self.ntl(F.interpolate(self.fake_B, size=(64,64), mode='bilinear', align_corners=False))
                self.logits_real_B = self.ntl(F.interpolate(self.real_B, size=(64,64), mode='bilinear', align_corners=False))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):  # bidirectional
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        lambda_confidence = self.opt.confidence_weight
        lambda_balance = self.opt.class_balance_weight
        use_feedback, use_cycle = self.opt.GAN_structure
        epsilon = self.opt.grad_epsilon
        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        # data-intrinsic disguising
        # GAN loss D_A(G_A(A)), Eq.1 in the paper
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B)), Eq.5 in the paper (bidirectional)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)  # bidirectional
        # Feedback loss || G_B(G_A(A)) - A||, Eq.3 in the paper
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A  # feedback
        # Feedback loss || G_A(G_B(B)) - B||, Eq.6 in the paper (bidirectional)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B  # bidirectional
        # combined loss
        if [use_feedback, use_cycle] == [True, True]:  # full, Eq.7 in the paper
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        elif [use_feedback, use_cycle] == [True, False]:  # single direction
            self.loss_G = self.loss_G_A + self.loss_cycle_A
        elif [use_feedback, use_cycle] == [False, False]:  # single direction & no feedback
            self.loss_G = self.loss_G_A
        
        # model-guided disguising
        if self.ntl is not None:
            # calculate approximate gradient of guidance loss using finite difference
            self.grad_conf_B, self.grad_balance_B = guidance_grad(self.ntl, self.ntl_confidence_mae, self.ntl_class_mae, 
                                                                  self.logits_real_B, self.logits_fake_B, self.fake_B, epsilon)
            
            # clip gradients
            self.grad_conf_B = torch.clamp(self.grad_conf_B, min=-100, max=100)
            self.grad_balance_B = torch.clamp(self.grad_balance_B, min=-100, max=100)
            
            # convert gradients to loss
            self.loss_confidence_B = -torch.mean(self.grad_conf_B * self.fake_B) * lambda_confidence
            self.loss_balance_B = -torch.mean(self.grad_balance_B * self.fake_B) * lambda_balance
            
            # merge model-guided losses to final loss, Eq.14 in the paper
            self.loss_G += self.loss_confidence_B + self.loss_balance_B
            
        # Check gradients of ntl
        for name, param in self.ntl.named_parameters():
            if param.grad is not None:
                print(f"NTL: {name} has gradient:", param.grad is not None)
                ntl_grad = True
                exit()
        
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate gradients for D_B, bidirectional
        self.optimizer_D.step()  # update D_A and D_B's weights
