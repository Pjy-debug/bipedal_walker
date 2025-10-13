# TTA

## Create virtual env via conda

```bash
(base) ~$ conda create -n tta python=3.9
(base) conda activate tta
```

## Requirements

Note that `-n 1` and `-L 1` options are necessary to install packages one by one and treat every line in the `requirements.txt` file as a separate item.

```bash
(tta) ~$ cat requirements.txt | xargs -n 1 -L 1 pip install
```

- [Fiber](https://uber.github.io/fiber/)
- [NEAT-Python](https://neat-python.readthedocs.io/en/latest/installation.html)
- [OpenAI Gym](https://github.com/openai/gym)

Delete conda env

```bash
(tta) ~$ conda remove -n tta --all
```

List all conda envs

```bash
(tta) ~$ conda info --envs
```

## Run testing

```bash
(tta) ~$ python ne_test_paralell.py
```

## Run locally

To run locally on a multicore machine

```bash
(tta) ~$ ./run_poet_local.sh new_test
```

## Run on a Kubernetes cluster

Follow instructions [here](https://uber.github.io/fiber/advanced/#working-with-persistent-storage) to create a persistent volume.

Then run the following command:

```bash
(tta) ~$ ./run_poet_remote.sh new_test
```

To get the training logs:

```bash
(tta) ~$ fiber cp nfs:/persistent/logs/new_test .
(tta) ~$ fiber cp nfs:/persistent/logs/poet_new_test poet_new_test
```

## Run on a computer cluster

To containerize and run the code on a computer cluster (e.g., Google Kubernetes Engine on Google Cloud), please refer to [Fiber Documentation](https://uber.github.io/fiber/getting-started/#containerize-your-program).

## Run original version

Use this [legacy branch](https://github.com/uber-research/poet/tree/original_poet).

## References

+ [Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions](https://arxiv.org/abs/1901.01753)

+ [Enhanced POET: Open-Ended Reinforcement Learning through Unbounded Invention of Learning Challenges and their Solutions](https://arxiv.org/abs/2003.08536)

+ [Uber Engineering Blog describing POET](https://eng.uber.com/poet-open-ended-deep-learning/)

