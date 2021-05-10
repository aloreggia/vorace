# VORACE: Voting with Random Classifiers

In many machine learning scenarios, looking for the best classifier that fits a particular dataset can be very costly in terms of time and resources. Moreover, it can require deep knowledge of the specific domain.
We propose a new technique which does not require profound expertise in the domain and avoids the commonly used strategy of hyper-parameter tuning and model selection. Our method is an innovative ensemble technique that uses voting rules over a set of randomly-generated classifiers.
Given a new input sample, we interpret the output of each classifier as a ranking over the set of possible classes.
We then aggregate these output rankings using a voting rule, which treats them as preferences over the classes. 

## Code setup
The requirements.txt file includes the comprehensive list of required Python libraries.
Install the requirements using:

```
pip install -r requirements.txt
```

## Paper
**Voting with Random Classifiers (VORACE): Theoretical and Experimental Analysis**
* [arXiv preprint](https://arxiv.org/abs/1909.08996)

### How to cite
```
@inproceedings{Cornelio2020VotingWR,
  title={Voting with Random Classifiers (VORACE): Theoretical and Experimental Analysis},
  author={Cristina Cornelio and Michele Donini and Andrea Loreggia and M. Pini and F. Rossi},
  booktitle={JAAMAS},
  year={2021}
}
```

## License

* License type: **Apache 2.0**
* more info [here](LICENSE)