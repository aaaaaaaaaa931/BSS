# Enhancing Adversarial Transferability via Component-Wise Transformation

Paper in ([Enhancing Adversarial Transferability through Block Stretch and Shrink](https://arxiv.org/html/2511.17688))

Set up the environment by following the [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) instructions.  

**Place** the `bss.py` file in the `TransferAttack/transferattack/input_transformation` directory.  

**Edit** the `TransferAttack/transferattack/__init__.py` file to include:  

```python
'bss': ('.input_transformation.bss', 'BSS'),
```

#### Attack Command:
```bash
python main.py \
  --input_dir ./path/to/data \
  --output_dir adv_data/bss/resnet18 \
  --attack bss \
  --model resnet18
```

#### Evaluation Command:
```bash
python main.py \
  --input_dir ./path/to/data \
  --output_dir adv_data/bss/resnet18 \
  --eval
```

---

##  Citation

If this repository helps your research, please cite:
```bibtex
@misc{liu2025enhancingadversarialtransferabilityblock,
      title={Enhancing Adversarial Transferability through Block Stretch and Shrink}, 
      author={Quan Liu and Feng Ye and Chenhao Lu and Shuming Zhen and Guanliang Huang and Lunzhe Chen and Xudong Ke},
      year={2025},
      eprint={2511.17688},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.17688}, 
}
```

## Credits
This work is built upon [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) and [tf_to_pytorch_model](https://github.com/ylhz/tf_to_pytorch_model?tab=readme-ov-file). Many thanks to the original authors for their contributions!  
