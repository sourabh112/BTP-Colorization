import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import Generator, Discriminator
import utils
import argparse
import os
from logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='train', help='input dataset')
parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=25, help='train batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=100, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
data_dir = 'D:/btp2/pix2pix/' + params.dataset + '/'
save_dir = params.dataset + '_results/'
model_dir = params.dataset + '_model/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Data pre-processing
transform = transforms.Compose([transforms.Resize(params.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.5)])

#Data loading
    # Train data
train_data = DatasetFromFolder(data_dir, subfolder_input='C',subfolder_target='D', direction=params.direction, transform=transform,
                               resize_scale=params.resize_scale, crop_size=params.crop_size, fliplr=params.fliplr)
train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=params.batch_size,
                                                shuffle=True)

    # Test data
test_data = DatasetFromFolder(data_dir, subfolder_input='C',subfolder_target='D', direction=params.direction, transform=transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=1,
                                               shuffle=False)
test_input, test_target = test_data_loader.__iter__().__next__()

# Models
G = Generator(1, params.ngf, 3)
D = Discriminator(4, params.ndf, 1)
G.cuda()
D.cuda()
# G.normal_weight_init(mean=0.0, std=0.02)
# D.normal_weight_init(mean=0.0, std=0.02)

G.load_state_dict(torch.load(model_dir + '25generator_param.pkl'))
D.load_state_dict(torch.load(model_dir + '25discriminator_param.pkl'))

# Set the logger
D_log_dir = save_dir + 'D_logs'
G_log_dir = save_dir + 'G_logs'
if not os.path.exists(D_log_dir):
    os.mkdir(D_log_dir)
D_logger = Logger(D_log_dir)

if not os.path.exists(G_log_dir):
    os.mkdir(G_log_dir)
G_logger = Logger(G_log_dir)

# Loss function
BCE_loss = torch.nn.BCELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2))
D_optimizer = torch.optim.Adam(D.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))

# Training GAN
D_avg_losses = []
G_avg_losses = []

step = 0
for epoch in range(26, params.num_epochs):
    torch.cuda.empty_cache()
    D_losses = []
    G_losses = []

    # training
    for i, (input, target) in enumerate(train_data_loader):
        # print("!!!!Working!!!!!")
        # input & target image data
        x_ = Variable(input.cuda())
        y_ = Variable(target.cuda())

        # print("!!!!Working2!!!!!")
        # Train discriminator with real data
        D_real_decision = D(x_, y_).squeeze()
        real_ = Variable(torch.ones(D_real_decision.size()).cuda())
        D_real_loss = BCE_loss(D_real_decision, real_)

        # print("!!!!Working3!!!!!")
        # Train discriminator with fake data
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        fake_ = Variable(torch.zeros(D_fake_decision.size()).cuda())
        D_fake_loss = BCE_loss(D_fake_decision, fake_)


        # print("!!!!Working4!!!!!")
        # Back propagation
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()


        # print("!!!!Working5!!!!!")
        # Train generator
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, real_)


        # print("!!!!Working6!!!!!")
        # L1 loss
        l1_loss = params.lamb * L1_loss(gen_image, y_)


        # Back propagation
        G_loss = G_fake_loss + l1_loss
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()


        # loss values
        D_losses.append(D_loss.data)
        G_losses.append(G_loss.data)

        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
              % (epoch+1, params.num_epochs, i+1, len(train_data_loader), D_loss.data, G_loss.data))

        # ============ TensorBoard logging ============#
        D_logger.scalar_summary('losses', D_loss.data, step + 1)
        G_logger.scalar_summary('losses', G_loss.data, step + 1)
        step += 1

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    # Show result for test image
    gen_image = G(Variable(test_input.cuda()))
    gen_image = gen_image.cpu().data
    utils.plot_test_result(test_input, test_target, gen_image, epoch, save=True, save_dir=save_dir)

    #saving model per epoch
    torch.save(G.state_dict(), model_dir + str(epoch) + 'generator_param.pkl')
    torch.save(D.state_dict(), model_dir + str(epoch) + 'discriminator_param.pkl')

# Plot average losses
utils.plot_loss(D_avg_losses, G_avg_losses, params.num_epochs, save=True, save_dir=save_dir)

# Make gif
utils.make_gif(params.dataset, params.num_epochs, save_dir=save_dir)

# Save trained parameters of model
torch.save(G.state_dict(), model_dir + 'generator_param.pkl')
torch.save(D.state_dict(), model_dir + 'discriminator_param.pkl')
