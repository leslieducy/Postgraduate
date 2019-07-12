import matplotlib.pyplot as plt
import pandas as pd

class ResultDeal(object):
    def __init__(self, title="DQN"):
        self.title = title

    def plotIncome(self, income_list):
        plt.figure(figsize=(8,6), dpi=80)
        plt.title(self.title)
        plt.plot(income_list, label="precision")
        plt.xlabel("train(num)")
        plt.ylabel("money")
        plt.legend(loc='upper right')
        # plt.ylim(0,100)
        plt.show()

    def plotWandering(self, wandering_list):
        plt.figure(figsize=(8,6), dpi=80)
        plt.title(self.title)
        plt.plot(wandering_list, label="precision")
        plt.xlabel("train(num)")
        plt.ylabel("wandering time")
        plt.legend(loc='upper right')
        # plt.ylim(0,100)
        plt.show()
    # 结果存入csv文件中，方便后续绘图
    def saveCSV(self, data={"income":[1]}):
        test=pd.DataFrame(data=data)
        test.to_csv(self.title+'.csv', encoding='utf-8')