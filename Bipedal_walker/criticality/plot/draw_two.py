import matplotlib.pyplot as plt

def draw_roc(x,y, save_path='new_log/result/cur_roc_789.png'):
    plt.plot(x,y)
    plt.plot([0,1],[0,1],linestyle='dashed')
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    #plt.legend(loc=1)
    plt.xlim((-0.01,1.01))
    plt.ylim((-0.01,1.01))
    plt.show()
    plt.savefig(save_path)

def draw_pr(x,y, save_path='new_log/result/cur_pr_789.png'):
    y_new = []

    y_new=y.copy()
    # # 不知道这里在干什么
    # for k in range(len(y)):
    #     item = y[k]
    #     if 8 < k < len(y)-1:
    #         y_new.append(item + 0.1)
    #     else:
    #         y_new.append(item)
    plt.plot(x,y_new)
    plt.plot([0,1],[1,0],linestyle='dashed')
    plt.title('Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.legend(loc=1)
    plt.xlim((-0.01,1.01))
    plt.ylim((-0.01,1.01))
    plt.show()
    plt.savefig(save_path)

TPR_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9930555555555556, 0.9583333333333334, 0.8888888888888888, 0.8402777777777778, 0.7569444444444444, 0.6180555555555556, 0.5, 0.2986111111111111]
FPR_list=[0.9236111111111112, 0.875, 0.8194444444444444, 0.7569444444444444, 0.7291666666666666, 0.7083333333333334, 0.5763888888888888, 0.4861111111111111, 0.4375, 0.2916666666666667, 0.16666666666666666, 0.11805555555555555, 0.06944444444444445, 0.041666666666666664, 0.013888888888888888, 0.006944444444444444, 0.0, 0.0]
precison_list=[0.51985559566787, 0.5333333333333333, 0.549618320610687, 0.5691699604743083, 0.5783132530120482, 0.5853658536585366, 0.6343612334801763, 0.6728971962616822, 0.6956521739130435, 0.7741935483870968, 0.8562874251497006, 0.8903225806451613, 0.927536231884058, 0.952755905511811, 0.9819819819819819, 0.9888888888888889, 1.0, 1.0]

# TPR_list=[0.0,0.6,1.0]
# FPR_list=[1.0,0.7,0.0]
# precison_list=[1.0,0.8,0.6]

draw_roc(FPR_list, TPR_list, '../stage1/statics/roc_curve.png')
plt.clf()
draw_pr(TPR_list, precison_list, '../stage1/statics/pr_curve.png')
plt.clf()