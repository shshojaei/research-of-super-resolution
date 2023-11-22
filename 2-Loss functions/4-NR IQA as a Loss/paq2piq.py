!pip install pyiqa

####### Train part of ESRT code (https://github.com/luissen/ESRT) ######

model = ESRT(upscale = args.scale) #architecture.IMDN(upscale=args.scale)

l1_criterion = nn.L1Loss()
criterion_quality = pyiqa.create_metric('paq2piq', as_loss=True)

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)
    criterion_quality = criterion_quality.to(device)

### ... ###

def train(epoch):
    model.train()
    adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)

        loss_l1 = l1_criterion(sr_tensor, hr_tensor)

        loss_quality = torch.abs(1-(criterion_quality(sr_tensor)/100)).sum().mean(dim=0) #loss over a batch

        loss_sr = loss_l1 + (0.0001 * loss_quality)

        loss_sr.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_sr: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss_sr.item()))
