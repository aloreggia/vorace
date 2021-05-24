# VORACE: VOting with RAndom ClassifiErs

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
* [Autonomous Agents and Multi-Agent Systems](https://link.springer.com/article/10.1007/s10458-021-09504-y)

### How to cite
```
@article{Cornelio2021,
  author = {Cornelio, Cristina and Donini, Michele and Loreggia, Andrea and Pini, Maria Silvia and Rossi, Francesca},
  doi = {10.1007/s10458-021-09504-y},
  issn = {1573-7454},
  journal = {Autonomous Agents and Multi-Agent Systems},
  number = {2},
  pages = {22},
  title = {{Voting with random classifiers (VORACE): theoretical and experimental analysis}},
  url = {https://doi.org/10.1007/s10458-021-09504-y},
  volume = {35},
  year = {2021}
}
```

## License

* License type: **Apache 2.0**
* more info [here](LICENSE)