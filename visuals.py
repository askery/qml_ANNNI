import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import os

import itertools

# ============================================================
# ########################################################## #
# ============================================================

# interval 1: x in [0, 1/2]
def isi_(x):
    return (1-x)/x * ( 1 - (   (1-3*x+4*x**2)/(1-x) )**(1/2) )

# interval 2: x in (1/2,3/2]
def cic_(x):
    return 1.05 * (x - 0.5 )

# interval 3: x in (1/2,3/2]
def bkt_(x):
    return 1.05 * ( (x - 0.5 )*(x-0.1) )**(1/2)

# interval 4: nonnull
def pem_(x):
    return -x + 1/(4*x)

# ============================================================
# ########################################################## #
# ============================================================

def plot_all_analytical(inti, intc, intb, intp,
                        isi, bkt, cic, pem):
    
    # Plot All
    plt.plot(inti,isi, '-' ,  label = 'Ising')
    plt.plot(intb,bkt, '-^',  label = 'BKT')
    plt.plot(intc,cic, '-.',  label = 'CIC')
    plt.plot(intp,pem, '--',  label = 'Peschel-Emery')
    
    plt.xlabel('$\kappa$',   fontsize = 20)
    plt.ylabel('$g$',        fontsize = 20)
    
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(loc = 'best', framealpha=0.0)
    
    plt.show()



# ============================================================
# ########################################################## #
# ============================================================

def plot_all(inti, intc, intb, intp, isi, bkt, cic, pem,
             fil1, fil2, fil3, fil4, 
             plot_intermediates=False, print_stuff=False):
    
    # used in the name of the figures files
    
    filters_str = fil1+fil2+fil3
    if fil4:
        filters_str += f"_{fil4}"
    
    here = os.getcwd()

    path = here + "\\results\\"
    dirs = ['figs','outs', 'figs\\final_figs', 'figs\\aux_figs']
    for d in dirs:
        os.makedirs(path+d, exist_ok=True) 

    # files names
    names = os.listdir(path)

    # ======================================================

    # filters and stuff

    names = list(filter(lambda x: fil1 in x, names)) 
    names = list(filter(lambda x: fil2 in x, names)) 
    names = list(filter(lambda x: fil3 in x, names))
    if fil4:
        names = list(filter(lambda x: fil4 in x, names))

    # names = list(filter(lambda x: "None" in x, names))

    # separate clas vs quan
    clas  = list(filter(lambda x: 'CLASSICAL' in x, names))
    quan  = list(filter(lambda x: 'QUANTUM' in x, names))

    cond  = ['2_2', '3_2', '3_3', '4_2', '4_3']

    clasd = {}
    quand = {}

    for c in cond:
        clasd[c] = list(filter(lambda x: c in x, clas))
        quand[c] = list(filter(lambda x: c in x, quan))

    # ======================================================

    # CFG
    # array for kappa values
    j = np.round(np.linspace(0,1,11),2) 

    # ======================================================
    # ======================================================
    # ======================================================

    # For CLASSICAL Solutions
    g_clas = {}
    for c in cond:

        if print_stuff:
            print(c)
            
        i       = 0
        betac   = []
        prob0_c = []
        prob1_c = []
        
#         print(*clasd[c], sep="\n")
        
        if len(clasd[c]) != len(j):
            str_error = ""
            str_error += "\nNot all files were generated!!!\n"
            str_error += "\nFiles available:\n"
            str_error += "\n".join(clasd[c]) + "\n\n"
            str_error += f"Conds: {cond}"
            str_error += "\n\nPlease generate all files and try again."
            
            #raise ValueError(str_error)
            
            continue

        for name in clasd[c]:

            data = pd.read_csv(path+name, header = 0)

            #print(data)
            # plot conf

            plt.figure()
            plt.tick_params(axis='both', labelsize=15)
            plt.xlabel('g', fontsize = 20)
            plt.ylabel('Class probability', fontsize = 20)
            plt.tight_layout()

            betatest = data['g']
            prob1    = data['predict_proba_y=1']
            prob0    = 1 - prob1
            aux      = prob0 - prob1 #abs ( prob0 - prob1 )

            betac.append (betatest [np.argmin(aux)])
            prob0_c.append (prob0 [np.argmin(aux)])
            prob1_c.append (prob1 [np.argmin(aux)])

            #print( j[i], betatest [np.argmin(aux)], prob0[np.argmin(aux)], prob1[np.argmin(aux)] )

            # main PLOT
            plt.plot(betatest, prob0, label = '0')
            plt.plot(betatest, prob1, label = '1')
            plt.legend(loc = 'best')

            title = 'Classical'+'_'+ c + ' for $\kappa$ = ' +str(j[i]) + '\n' + \
                    'Training with $\kappa$ = 0.0' 
            plt.title(title)

            figname = 'classical_'+c+'_k_'+str(j[i])+filters_str+'.png'
            plt.savefig(path+'figs/aux_figs/'+figname, bbox_inches='tight')

            if plot_intermediates:
                plt.show()
            else:
                plt.close()

            i+=1

        g_clas[c]  = betac
        out   = {'k':j, 'g':g_clas[c]} 
        oname = 'classical_'+c+filters_str+'.csv'

        df    = pd.DataFrame(out)
        df.to_csv(path+'outs/'+oname, header = None, index=False) 

        if print_stuff:
            print(df)
        print(oname, 'SAVED')

        print('='*50)

    # ======================================================

    plt.plot(inti,isi, '-' ,  label = 'Ising') #uncomment for analytical sol
    plt.plot(intb,bkt, '-^',  label = 'BKT')   #uncomment for analytical sol

    for m,c in enumerate(cond):

        plt.xlabel('$\kappa$',   fontsize = 20)
        plt.ylabel('$g$',        fontsize = 20)
        plt.tick_params(axis='both', labelsize=15)
        plt.plot(j,g_clas[c], marker = m, label = 'clas_'+c)

        #plt.legend(loc = 'best', framealpha=0.0)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

        title = 'classical_phase_diagram'+filters_str
        name  = title+'.png'

        #name = 'classical_phase_diagram'+filters_str+'.png'
        plt.title(title)
        plt.savefig(path+'figs/final_figs/'+name, bbox_inches='tight')
        
    plt.show()

    # ======================================================
    # ======================================================
    # ======================================================

    # for QUANTUM solutions
    g_quan = {}
    for c in cond:

        if print_stuff:
            print(c)
            
        i       = 0
        betac   = []
        prob0_c = []
        prob1_c = []

        for name in quand[c]:

            data = pd.read_csv(path+name, header = 0)
            #print(data)
            # plot conf

            plt.figure()
            plt.tick_params(axis='both', labelsize=15)
            plt.xlabel('g', fontsize = 20)
            plt.ylabel('Class probability', fontsize = 20)
            plt.tight_layout()

            #
            betatest = data['g']
            prob1    = data['predict_proba_y=1']
            prob0    = 1 - prob1
            aux      = abs ( prob0 - prob1 )

            betac.append (betatest [np.argmin(aux)])
            prob0_c.append (prob0 [np.argmin(aux)])
            prob1_c.append (prob1 [np.argmin(aux)])

            #print( j[i], betatest [np.argmin(aux)], prob0[np.argmin(aux)], prob1[np.argmin(aux)] )

            # main PLOT
            plt.plot(betatest, prob0, label = '0')
            plt.plot(betatest, prob1, label = '1')
            plt.legend(loc = 'best')
            title = 'Quantum'+'_'+ c + ' for $\kappa$ = ' +str(j[i]) + '\n' + \
                    'Training with $\kappa$ = 0.0' 
            plt.title(title)
            figname = 'quantum_'+c+'_k_'+str(j[i])+filters_str+'.png'
            plt.savefig(path+'figs/aux_figs/'+figname, bbox_inches='tight')

            if plot_intermediates:
                plt.show()
            else:
                plt.close()

            i+=1

        g_quan[c]  = betac

        out   = {'k':j, 'g':g_quan[c]} 
        oname = 'quantum_'+c+filters_str+'.csv'

        df    = pd.DataFrame(out)
        df.to_csv(path+'outs/'+oname, header = None, index=False) 

        if print_stuff:
            print(df)
        print(oname, 'SAVED')
        
        print('='*50)

    # ======================================================

    plt.plot(inti,isi, '-' ,  label = 'Ising') #uncomment for analytical sol
    plt.plot(intb,bkt, '-^',  label = 'BKT')   #uncomment for analytical sol

    for m,c in enumerate(cond):

        plt.xlabel('$\kappa$',   fontsize = 20)
        plt.ylabel('$g$',        fontsize = 20)
        plt.tick_params(axis='both', labelsize=15)
        plt.plot(j,g_quan[c], marker = m, label = 'quan_'+c)

        #plt.legend(loc = 'best', framealpha=0.0)

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

        title = 'quantum_phase_diagram'+filters_str
        name  = title+'.png'

        plt.title(title)
        plt.savefig(path+'figs/final_figs/'+name, bbox_inches='tight')

    plt.show()
    
# ============================================================
# ########################################################## #
# ============================================================

def iterate_plots(df_combs,
                  inti, intc, intb, intp, isi, bkt, cic, pem,
                  plot_intermediates=False, print_stuff=False):
    
    for row in range(df_combs.shape[0]):
        
        fil1, fil2, fil3, fil4 = df_combs.loc[row, :]
        
        print(f"Row: {row}")
        print(f"Feature selection strategy: {fil1.strip('_')}")
        print(f"Desambiguation strategy: {fil2.strip('_')}")
        print(f"Discretization strategy: {fil3.strip('_')}")
        if fil4:
            print(f"Number of principal components considered: {fil4.strip('_')}")
        print()

        plot_all(inti, intc, intb, intp, isi, bkt, cic, pem,
                 fil1, fil2, fil3, fil4, 
                 plot_intermediates, print_stuff)
        
        print()
        print("#"*80)
        print()
        
# ============================================================
# ########################################################## #
# ============================================================


# ============================================================
# ########################################################## #
# ============================================================