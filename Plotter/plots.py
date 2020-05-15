import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def integrate(x, y):
    area = np.trapz(y=y, x=x)
    return area


dataset = 'fair-movielens'
result_path = '{0}_metrics10_best0.tsv'.format(dataset)

data = pd.read_csv(result_path, sep='\t')

recommender_name = {'bprmf': 'BPR-MF', 'apr': 'AMF', 'random': 'Random'}
metric_name = {'EFD_Top10': 'EFD@10', 'EPC_Top10': 'EPC@10', 'Gini_Top10': 'Gini@10', 'ICov_Top10': 'ICov@10',
               'Prec_Top10': 'P@10', 'Recall_Top10': 'R@10', 'SE_Top10': 'SE@10', 'UCov': 'UCov',
               'nDCG_Top10': 'nDCG@10'}
attack_name = {}

auc_values = {}

for dataset in ['fair-movielens', 'lastfm']:

    auc_values.setdefault(dataset, {})
    for recommender in ['bprmf', 'apr']:

        auc_values[dataset].setdefault(recommender, {})
        rec_data = data[data.recommender.astype(str).str.contains('^{0}'.format(recommender))]

        for metric in data.columns[7:]:
            auc_values[dataset][recommender].setdefault(metric_name[metric], {})
            plt.ylim(0, 1)

            plt.figure()

            # Baseline Recommender
            val = float(data[data.Algorithm.astype(str).str.contains('^{0}'.format(recommender))][metric].iloc[0])
            plt.hlines(val, 0, 500, linestyles='solid', color='r', label=recommender_name[recommender])

            # Random Recommender
            plt.hlines(float(data[data.Algorithm.astype(str).str.contains('^{0}'.format('random'))][metric].iloc[0]), 0,
                       500,
                       linestyles='solid', color='m', label=recommender_name['random'])

            # FGSM eps = 0.5
            plt.hlines(float(rec_data[(rec_data.attack == 'fgs') & (rec_data.eps == 0.5)][metric]), 0, 500,
                       linestyles='solid', color='c', label=r'FGSM ($\epsilon=0.5$)')

            # FGSM eps = 1.0
            plt.hlines(float(rec_data[(rec_data.attack == 'fgs') & (rec_data.eps == 1.0)][metric]), 0, 500,
                       linestyles='dashed', color='c', label=r'FGSM ($\epsilon=1.0$)')

            # BIM 500 eps = 0.5
            x = np.array(rec_data[(rec_data.attack == 'bim') & (rec_data.eps == 0.5)].sort_values(by='iteration')[
                             'iteration'].to_list())
            y = np.array(rec_data[(rec_data.attack == 'bim') & (rec_data.eps == 0.5)].sort_values(by='iteration')[
                             metric].to_list())
            # AUC Val
            auc_values[dataset][recommender][metric_name[metric]]['bim0.5'] = integrate(x, y)
            # Plot
            x_new = np.linspace(x.min(), x.max(), 500)
            f = interp1d(x, y, kind='quadratic')
            y_smooth = f(x_new)
            plt.plot(x_new, y_smooth, color='b', label=r'BIM ($\epsilon=0.5$)')

            # BIM 500 eps = 1.0
            x = np.array(rec_data[(rec_data.attack == 'bim') & (rec_data.eps == 1.0)].sort_values(by='iteration')[
                             'iteration'].to_list())
            y = np.array(rec_data[(rec_data.attack == 'bim') & (rec_data.eps == 1.0)].sort_values(by='iteration')[
                             metric].to_list())
            # AUC Val
            auc_values[dataset][recommender][metric_name[metric]]['bim1.0'] = integrate(x, y)
            x_new = np.linspace(x.min(), x.max(), 500)
            f = interp1d(x, y, kind='quadratic')
            y_smooth = f(x_new)
            plt.plot(x_new, y_smooth, 'b--', label=r'BIM ($\epsilon=1.0$)')

            # PGD 500 eps = 0.5
            x = np.array(rec_data[(rec_data.attack == 'pgd') & (rec_data.eps == 0.5)].sort_values(by='iteration')[
                             'iteration'].to_list())
            y = np.array(rec_data[(rec_data.attack == 'pgd') & (rec_data.eps == 0.5)].sort_values(by='iteration')[
                             metric].to_list())
            # AUC Val
            auc_values[dataset][recommender][metric_name[metric]]['pgd0.5'] = integrate(x, y)
            x_new = np.linspace(x.min(), x.max(), 500)
            f = interp1d(x, y, kind='quadratic')
            y_smooth = f(x_new)
            plt.plot(x_new, y_smooth, color='g', label=r'PGD ($\epsilon=0.5$)')

            # PGD 500 eps = 1.0
            x = np.array(rec_data[(rec_data.attack == 'pgd') & (rec_data.eps == 1.0)].sort_values(by='iteration')[
                             'iteration'].to_list())
            y = np.array(rec_data[(rec_data.attack == 'pgd') & (rec_data.eps == 1.0)].sort_values(by='iteration')[
                             metric].to_list())
            # AUC Val
            auc_values[dataset][recommender][metric_name[metric]]['pgd1.0'] = integrate(x, y)
            x_new = np.linspace(x.min(), x.max(), 500)
            f = interp1d(x, y, kind='quadratic')
            y_smooth = f(x_new)
            plt.plot(x_new, y_smooth, 'g--', label=r'PGD ($\epsilon=1.0$)')

            plt.xlabel('# Iteration')
            plt.ylabel(metric_name[metric])

            handles, labels = plt.gca().get_legend_handles_labels()
            order = [4, 5, 6, 7, 0, 1, 2, 3]
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='best')
            plt.grid()

            plt.savefig('{0}/{1}.png'.format(dataset, metric), quality=100)

print('Data\tRec\tMetric\tBIM0.5\tBIM1.0\tPGD0.5\tPGD1.0')
for dataset in auc_values.keys():
    for recommender in auc_values[dataset].keys():
        for metric in auc_values[dataset][recommender].keys():
            print('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format(
                dataset,
                recommender,
                metric,
                auc_values[dataset][recommender][metric]['bim0.5'],
                auc_values[dataset][recommender][metric]['bim1.0'],
                auc_values[dataset][recommender][metric]['pgd0.5'],
                auc_values[dataset][recommender][metric]['pgd1.0'],
            ))
