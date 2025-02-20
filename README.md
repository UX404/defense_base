# Attack and Defense （Preview）
Attack and Defense API for adversarial training project for Machine Learning in Action course in SJTU 


### Set up:
** Requirement ** `python3`, `pytorch > 1.2`, `tqdm`  
1. Install python  
Recomand install anaconda, see https://www.anaconda.com/products/individual#Downloads
2. Create new environment and install pytorch, tqdm:
```
conda create -n t17 python=3.8
conda install pytorch=1.7 torchvision cudatoolkit=10.2 -c pytorch
pip install tqdm
```
### Notes
1. About args  
代码中使用的是Python包`argparse`，来在命令行解析参数。例如设置attack的model: `python attack_main.py --model_name=model1`  
2. About gpu  
推荐使用GPU训练，多卡训练，需设置gpu_id, eg: `python pgd_train.py --gpu_id=0,1`.
在这里使用 Pytorch 的nn.DataParallel 实现多卡并行， 涉及代码如下。nn.DataParallel 实际上 wrap the pytorch model as `model.module`.
```
device = torch.device('cuda')
model = ResNet18().to(device)    
model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
```
3. About log  
关于日志保存，请参见代码中`Logger` class, 也可自行实现日志部分  
4. 其他问题我会更新在这里  


# Attack Task
In this task, you need to design attack algorithm to attack provided 6 models.  
Note that we use constraint of `l-inf` norm distance `< 8./255`. 

### Dataset: CIFAR10
Use `prepare_cifar` in utils.py to get `train_loader` and `test_loader` of CIFAR10.  

```python
from utils import prepare_cifar
train_loader, test_loader = prepare_cifar(batch_size = 128, test_batch_size = 256)
```

### Defense models
1. model1:  vanilla resnet34
2. model2:  PGD adversarial trained resnet18
3. model3:  Unkonown resnet
4. model4:  [TRADES](https://arxiv.org/abs/1901.08573)
5. model5:  [PGD_HE](https://arxiv.org/abs/2002.08619)
6. model6:  [RST_AWP](https://arxiv.org/abs/2004.05884)

Step1, download model weights here [Jbox](https://jbox.sjtu.edu.cn/l/PFOOnZ)  
Step2, move model weights to path `models/weights/`  
Run model: see `get_model_for_attack` in model.py  

### Attack Baseline
See PGDAttack class in pgd_attack.py for PGD attack baseline code.  
You can modify attack function in attack_main.py to your attack class.  

### 代码提交规范
你需要像pgd_attack.py 里的PGDAttack class一样实现一个攻击类。 类似： 
```python
attack = PGDAttack(args.step_size, args.epsilon, args.perturb_steps)
```
然后可以通过这样调用攻击算法,生成对抗样本：
```python
x_adv = attack(model, x, label)
```
最后，我们会使用 `eval_model_with_attack` in eval_model.py 来测试你的攻击算法。 
```python
natural_acc, robust_acc, distance = eval_model_with_attack(model, test_loader, attack, device)
```

### Test your attack
See attack_main.py, and replace pgd_attack method to your own attack method.   
And run attack_main.py to test your attack, set model_name to [model1, model2, model3, model4, model5, model6]. Like:  
```sh
python attack_main.py --model_name=model1
```



# Defense Task
In this task, you need to train a robust model under l-inf attack(8/255) on CIFAR10.  

### Evaluate your model
We'll use various attack methods to evaluate the robustness of your model.   
Include PGD attack, and others.  

### How to run PGD attack to test robustness of your model
1. Open attack_main.py, specify how to load your model.  
  A example in attack_main.py:
```python
# Change to your model and specify how to load your model here
model =  WideResNet().to(device)  
model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
model.load_state_dict(torch.load(args.model_path)['state_dict'], strict=False)
```
2. Run attack_main.py:  
```python
python attack_main.py --model_path=your_weight_path --gpu_id=0      # eg. For multiple gpus, set --gpu_id=1,2,3
```
It will give natural acc and robust acc for your model.

### A Defense Baseline: Adversarial training   
See pgd_train.py 

### 代码提交规范
你需要提供模型的训练，推理方法和模型weight。请参考本repo提供defense model的方法。提交的代码包必须包含:
```
train.py   # python train.py 可直接复现你的模型训练过程
infer.py   # python inter.py  可直接测试 cifar10 test acc，给出natural acc.
attack.py  # python attack.py 运行攻击你的模型， 请参考attack_main.py 把被攻击模型换成你的。
your_model_name.pth  #你训练的模型weight
```
将以上文件打包压缩提交

