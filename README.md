# EO foundational model inference and fine tuning (self-directed training)

## Objective

The governing objective is to learn how to run inference on finely-tuned models, and to fine-tune a foundational model.  For this excercise
I have chosen to use the [IBM-NASA Prithvi Models Family](https://huggingface.co/ibm-nasa-geospatial)

A more fine-grained list of objectives follows:

## Checklist

- ✅ Understand foundational model architecture and capabilities
- ✅ Build a script to download HLS imagery from the Microsoft Planetary Computer archive
- [ ] Run inference on multiband HLS imagery for crop-coverage
  - ✅ Using Huggingface Docker file run locally
  - ❌ Using local Python environment
    - [ ] Need to update inference script to accommodate modern OpenMMLab API
  - [ ] Containerize new Python environment
  - [ ] Deploy onto cloud service with API
- [ ] Locally fine tune foundational model to predict 🤔
- [ ] Exlore could services to speed up fine tuning
