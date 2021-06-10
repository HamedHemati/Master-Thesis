import torch
import torch.autograd as autograd


def generate_synthetic_features(net_g, opt, classes, cls_embs, n_samples, cuda, multi_out=False):
    """Generates synthetic data for the eval/test stages
    """
    torch.set_grad_enabled(False)
    n_cls = classes.size(0)
    feat = torch.FloatTensor(n_cls*n_samples, opt.n_feat)
    label = torch.LongTensor(n_cls*n_samples) 
    cls_emb = torch.FloatTensor(n_samples, opt.n_cls_emb)
    if cuda:
        cls_emb = cls_emb.cuda()
    for i in range(n_cls):
        cls_emb.copy_(cls_embs[classes[i]].repeat(n_samples, 1))
        noise = torch.randn(n_samples, opt.n_z)
        if cuda:
            noise = noise.cuda()
        if multi_out:
            output, _ = net_g(noise, cls_emb)
        else:            
            output = net_g(noise, cls_emb) 
        feat.narrow(0, i*n_samples, n_samples).copy_(output.data.cpu())
        label.narrow(0, i*n_samples, n_samples).fill_(classes[i])
    torch.set_grad_enabled(True)
    return feat, label


def generate_synthetic_features_mean(net_g, opt, classes, cls_embs, n_samples, cuda):
    """Generates synthetic data for each class and return the mean
    """
    torch.set_grad_enabled(False)
    n_cls = classes.size(0)
    feat = torch.FloatTensor(n_cls, opt.n_feat)
    label = torch.LongTensor(n_cls) 
    cls_emb = torch.FloatTensor(n_samples, opt.n_cls_emb)
    if cuda:
        cls_emb = cls_emb.cuda()
    for i in range(n_cls):
        cls_emb.copy_(cls_embs[classes[i]].repeat(n_samples, 1))
        noise = torch.randn(n_samples, opt.n_z)
        if cuda:
            noise = noise.cuda()    
        output = net_g(noise, cls_emb) 
        feat[i] = torch.mean(output.data.cpu(), dim=0)
        label[i] = classes[i]
    torch.set_grad_enabled(True)
    return feat, label


def calc_gradient_penalty(net_d, opt, real, fake, cls_emb, batch_size, cuda):
    """Calculates gradient penaly for the discriminator
    """
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real.size())
    if cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real + ((1 - alpha) * fake)
    if cuda:
        interpolates = interpolates.cuda()
    interpolates.requires_grad = True
    cls_emb.requires_grad = False
    disc_interpolates = net_d(interpolates, cls_emb)
    #disc_interpolates, _ = net_d(interpolates, cls_emb)
    ones = torch.ones(disc_interpolates.size())
    if cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda_gp
    return gradient_penalty  