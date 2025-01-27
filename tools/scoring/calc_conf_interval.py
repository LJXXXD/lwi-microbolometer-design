
import numpy as np
import scipy.stats as stats



def calc_conf_interval(pred_list, targ_list, confidence=0.95):
    diff_list = pred_list - targ_list
    conf_interval = []
    # print('targ list len', len(targ_list))
    # print('targ len', len(targ_list[0]))
    for i in range(len(targ_list[0])):
        data = diff_list[:, i]

        # # Calculating the 2.5th and 97.5th percentiles
        # lower_quantile = np.percentile(data, 0.5 - confidence/2)
        # upper_quantile = np.percentile(data, 0.5 + confidence/2)





        ae, loce, scalee = stats.skewnorm.fit(data)
        dist = stats.skewnorm(ae, loce, scalee)
        lower_quantile = dist.ppf(0.5 - confidence/2)  # Lower quantile covering % of the data
        upper_quantile = dist.ppf(0.5 + confidence/2)  # Upper quantile covering % of the data
        interval = upper_quantile - lower_quantile
        conf_interval.append(interval)

    return conf_interval