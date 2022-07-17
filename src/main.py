from MDP import MDPClass
import numpy
from datetime import datetime

file_log = open('../logs/log'+datetime.now().strftime("%Y%m%d%H%M%S"),'w+')

ambientes = ["Ambiente1", "Ambiente2"]
X = [5, 20]
Y = [25, 100]

Acoes = numpy.array(['N','S','L','O'])

for i in range(len(X)):

    file_log.write("Ambiente:" + str(i)+"\n")
    mdp1 = MDPClass(X[i],Y[i], Acoes, ambientes[i])

    goals = mdp1.goalStates()

    t1 = datetime.now().strftime("%Y%m%d%H%M%S")
    pol1,v1 = mdp1.valueIterationMin2(0.000001,0.999999)
    t2 = datetime.now().strftime("%Y%m%d%H%M%S")
    AC1 = mdp1.acessosExecutados()

    file_log.write("T1:" + t1+"\n")
    file_log.write("T2:" + t2+"\n")
    file_log.write("AC1:" + str(AC1)+"\n")
    file_log.write("Pol1:\n")
    for p in pol1:
        file_log.write(str(p))
        file_log.write(", ")
    file_log.write("\n")
    file_log.flush()

    mdp1.zeraAcessos()
    t3 = datetime.now().strftime("%Y%m%d%H%M%S")
    pol2 = mdp1.LAO_star((X[i] * Y[i])//2,goals,0.000001, 0.999999)
    t4 = datetime.now().strftime("%Y%m%d%H%M%S")
    AC2 = mdp1.acessosExecutados()

    file_log.write("T3:" + t3+"\n")
    file_log.write("T4:" + t4+"\n")
    file_log.write("AC2:" + str(AC2)+"\n")
    file_log.write("Pol2:\n")
    for p in pol2:
        file_log.write(str(p))
        file_log.write(", ")
    file_log.write("\n")
    file_log.flush()

    mdp1.zeraAcessos()
    t5 = datetime.now().strftime("%Y%m%d%H%M%S")
    pol3 = mdp1.BLAO_star((X[i] * Y[i])//2,goals,0.999999, 0.000001, True)
    t6 = datetime.now().strftime("%Y%m%d%H%M%S")
    AC3 = mdp1.acessosExecutados()

    file_log.write("T5:" + t5+"\n")
    file_log.write("T6:" + t6+"\n")
    file_log.write("AC3:" + str(AC3)+"\n")
    file_log.write("Pol3:\n")
    for p in pol3:
        file_log.write(str(p))
        file_log.write(", ")
    file_log.write("\n")
    file_log.flush()    

file_log.close()
