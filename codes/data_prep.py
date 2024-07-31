
import pandas as pd, numpy as np, os, torch, sklearn as sk, time, re, random


def get_data(data1, data2):
    source = pd.read_csv(data1)
    target = pd.read_csv(data2)
    print(target.shape, source.shape)

    source.rename(columns = {"sample":"features"},inplace=True)
    target.rename(columns = {"sample":"features"},inplace=True)
    target.fillna(0, inplace=True)
    source.fillna(0, inplace=True)

    sfeat = set(source['features'])
    tfeat = set(target['features'])
    source_sample = set(source.columns[1:])
    target_sample = set(target.columns[1:])
    source.set_index("features", inplace=True)
    target.set_index("features", inplace=True)
    print({x:y for x,y in zip(["source features", "target features", "source samples", "target samples"],[len(sfeat), len(tfeat), len(source_sample), len(target_sample)])})
    train_samples = sorted(list(source_sample.intersection(target_sample)))
    infer_samples = sorted(source_sample.difference(target_sample))
    len(train_samples), len(infer_samples)

    trainXm = source[list(source_sample.intersection(target_sample))]
    inferXm = source[list(source_sample.difference(target_sample))]
    trainXmi = target[list(source_sample.intersection(target_sample))]

    train_samples = sorted(list(source_sample.intersection(target_sample)))
    infer_samples = sorted(source_sample.difference(target_sample))

    len(train_samples), len(infer_samples)
    data_names = "sourceXm trainXm trainXmi inferXm train_samples infer_samples source_feat target_feat".split()
    data_sets = [source, trainXm ,trainXmi ,inferXm ,train_samples ,infer_samples, len(sfeat), len(tfeat)]

    return {x:y for x,y in zip(data_names,data_sets)}
# %%


if __name__ == '__main__':
    
    print(get_data("../data/mRNA.csv", "../data/miRNA.csv").keys())