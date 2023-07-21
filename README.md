# L2L Challenge Example Submission
This is an example for how to prepare a valid submission for the MICCAI L2L Challenge

It is based on the [pretrained imagenet](https://github.com/StefanoWoerner/mimeta-pytorch/blob/master/examples/imagenet-pretrained.py)
example from [MIMeta](https://github.com/StefanoWoerner/mimeta-pytorch)
and process input images and generates predictions by fine-tuning a pretrained
network on the target task.

A real submission will likely be more complex and probably contain multiple python
modules, but this example should give you a good starting point.

## How to replicate this example

### Prerequisites
- Python 3.10
- pip
- singularity

### Install dependencies

Create a new environment with python>=3.10 and install the dependencies:
```bash
pip install -r requirements.txt
```

### Download the model
In the `example_submission` directory, run `download.py` to download the model. This
allow us to put the model in the container and avoid downloading it at evaluation time.
Containers will not have access to the internet on the evaluation server.

```bash
python download.py
```

### Build the singularity container

Still in the `example_submission` directory, run the following command to build the
singularity container:
```bash
singularity build --fakeroot --force example-submision.sif singularity.def
```

### Submit

- Create an account on the [L2L challenge portal](https://portal.l2l-challenge.org/)
  and subsequently a team.
- Go to the [submission page](https://portal.l2l-challenge.org/submission) and submit
  the `example-submision.sif` file.


## Authors
Stefano Woerner, Bartłomiej Baranowski
