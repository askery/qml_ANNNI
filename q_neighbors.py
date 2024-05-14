import numpy as np
import pandas as pd

from scipy.linalg import expm

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute

from qiskit.circuit import AncillaQubit

from qiskit.visualization import plot_histogram

from qiskit.circuit.library.standard_gates import XGate, ZGate, HGate

from qiskit.circuit.add_control import add_control

from qiskit.extensions import UnitaryGate

import matplotlib.pyplot as plt

#######################################################################################
#######################################################################################

def show_figure(fig):
    '''
    auxiliar function to display plot 
    even if it's not the last command of the cell
    from: https://github.com/Qiskit/qiskit-terra/issues/1682
    '''
    
    new_fig = plt.figure()
    new_mngr = new_fig.canvas.manager
    new_mngr.canvas.figure = fig
    fig.set_canvas(new_mngr.canvas)
    plt.show(fig)
    
#######################################################################################
#######################################################################################

def test_circuit(qc):
    ''' 
    auxiliar function, to test intermediate steps, looking at measurement results
    this function allows the inspection of the amplitudes of states, 
    at any point of the circuit (call it with the desired circuit)
    '''
    
    cr = ClassicalRegister(qc.num_qubits, "creg_test")
    qc_cr = QuantumCircuit(cr)

    # qc_test = qc + qc_cr
    # new version: use tensor
    qc_test = qc.tensor(qc_cr)

    for i in range(qc.num_qubits):
        qc_test.measure(i, i)

    #################################
    
    backend = Aer.get_backend('aer_simulator')
    try:
        backend.set_options(device='GPU')
    except:
        pass

    # backend = Aer.get_backend("qasm_simulator")

    job = execute(qc_test, backend, shots=1e5, seed_simulator=42)
    results = job.result()
    counts = results.get_counts()

    return plot_histogram(counts, title="Results", figsize=(12, 4))


#######################################################################################
#######################################################################################

def construct_input_state(X_test):
    '''
    construct quantum state from bit string X_test
    '''

    # number of features
    n = len(X_test)

    # build the circuit
    
    qr_input = QuantumRegister(n, "features_input")
    qc_input = QuantumCircuit(qr_input)

    for i in range(n):
        if X_test[i] == "1":
            qc_input.x(i)
        
    return qc_input

#######################################################################################
#######################################################################################

def construct_training_set_state(X, y):
    '''
    construct quantum superposition of training dataset from:
    - X: feature matrix
    - y: target array
    returns the circuit as well as the number of features, n
    '''
    
    # number of examples
    N = X.shape[0]

    # number of features
    n = X.shape[1]

    # full dataset, of the form [[X], [y]]
    dataset = np.append(X, y.reshape((-1, 1)), axis=1)
    
    # n+1 because the last register will encode the class!
    amplitudes = np.zeros(2**(n+1))
    
    # integer representation of binary strings in training dataset
    # notice the [::-1], which is necessary to adjust the endianness!
    data_points_X = [int("".join(str(i) for i in X[j])[::-1], 2) for j in range(dataset.shape[0])]
    
    # integer representation considering also the class
    # IMPORTANT: sum 2**n for elements in class 1
    # this is necessary to encode also the class, in n+1 registers!
    data_points = [x + 2**n if y[i] == 1 else x for i, x in enumerate(data_points_X)]
    
    # set amplitudesof existing datapoints
    amplitudes[data_points] = 1
    
    # normalize amplitudes
    amplitudes = amplitudes/np.sqrt(amplitudes.sum())
    
    ###################################################
    
    # build the circuit
    
    qr_features = QuantumRegister(n, "features_train")
    qr_target = QuantumRegister(1, "target_train")

    qc_training = QuantumCircuit(qr_features, qr_target)

    # in previous versions, i used this, but it doesnt work anymore...
    try:
        qc_training.initialize(amplitudes, [qr_features, qr_target])
    # ...now, indices must be passed instead of quantum regs (apparently)
    except:
        qc_training.initialize(amplitudes, list(range(n+1)))
        
    return qc_training, n


#######################################################################################
#######################################################################################

def construct_ancilla_state():
    '''
    construct ancilla register
    '''
    
    # build the circuit
    
    qr_anc = QuantumRegister(1, "anc")
    qc_anc = QuantumCircuit(qr_anc)

    qc_anc.draw("mpl")
        
    return qc_anc

#######################################################################################
#######################################################################################

def construct_initial_state(X, y, X_test, draw=False):
    '''
    conjugate elements of the initial state (input, training dataset and ancilla) in a
    single quantum circuit
    returns the circuit as well as the number of features, n
    '''
    
    qc_input = construct_input_state(X_test)
    
    qc_training, n = construct_training_set_state(X, y)
    
    qc_anc = construct_ancilla_state()

    # qc = qc_input + qc_training + qc_anc
    # new version: use tensor
    qc = qc_anc.tensor(qc_training).tensor(qc_input)

    if draw:
        
        qc.barrier()
        
        print("\nInitial state:")
        show_figure(qc.draw("mpl"))
    
    return qc, n

#######################################################################################
#######################################################################################

def step1(qc, draw=False):
    
    qc.h(-1)

    if draw:
        
        qc.barrier()
            
        print("\nStep 1:")
        show_figure(qc.draw("mpl"))
        
    return qc

#######################################################################################
#######################################################################################

def step2(qc, n, draw=False):
    
    for i in range(n):

        qc.cnot(i, n+i)

    if draw:
        
        qc.barrier()
            
        print("\nStep 2:")
        show_figure(qc.draw("mpl"))
        
    return qc

#######################################################################################
#######################################################################################

def get_qubit_indices(qc):
    '''
    auxiliar function.
    
    returns a dictionary of the following form:
    
    {register_label : [indices of the qubits in register]}
    
    indices follow the order of appearance, as usual ;)
    
    this function is necessary now (newer versions) that qiskit is using indices instead of registers
    '''
    registers_sizes = [qr.size for qr in qc.qregs]
    registers_labels = [qr.name for qr in qc.qregs]

    # auxiliar list used below to define the ranges
    aux_count = [0] + np.cumsum(registers_sizes).tolist()

    q_indices = [list(range(aux_count[i], aux_count[i+1])) for i in range(len(aux_count)-1)]

    dict_q_indices = dict(zip(registers_labels, q_indices))

    return dict_q_indices

#######################################################################################
#######################################################################################

def step3(qc, n, draw=False):
    
    # matrices of 1 and \sigma_z to be exponentiated below
    # source of the matrix exponential methods below:
    # https://quantumcomputing.stackexchange.com/questions/10317/quantum-circuit-to-implement-matrix-exponential

    idtt = np.eye(2)

    sz = np.array([[1,0], 
                   [0,-1]])
    
    ###################################################
    
    # define the exponentiated unitaries
    U_1_minus = expm(-1j * (np.pi/(2*4)) * ((idtt-sz)/2))
    U_1_plus = expm(1j * (np.pi/(2*4)) * ((idtt-sz)/2))

    # defining controlled gates
    u1m = add_control(operation=UnitaryGate(U_1_minus, label="$U_1^-$"),
                      num_ctrl_qubits=1, ctrl_state=0, label="$CU_1^-$")

    u1p = add_control(operation=UnitaryGate(U_1_plus, label="$U_1^+$"),
                      num_ctrl_qubits=1, ctrl_state=1, label="$CU_1^+$")


    # getting the registers
    registers = qc.qregs
    
    # register labels
    registers_labels = [qr.name for qr in qc.qregs]

    qr_features = registers[registers_labels.index("features_train")]
    qr_anc = registers[registers_labels.index("anc")]
    
    # build a circuit to apply the unitary above
    # this will be combined with the main circuit later on (notice: same registers!!).
    qc_u = QuantumCircuit(qr_features, qr_anc)

    for i in range(n):

        # apply the unitaries
        qc_u.append(u1p, [qr_anc[0], qr_features[i]])
        qc_u.append(u1m, [qr_anc[0], qr_features[i]])
    
    ###################################################
    
    # combine the U circuit above to the main circuit
    # qc = qc.combine(qc_u)
    
    # new: in new version, use "compose" instead of combine
    # to compose properly, we must pass the qubit indices (!!). in order to make
    # the code as general as possible, I had to use the code below7
    # (and introduce the aux function get_qubit_indices, above)
    
    qc_u_qregs_labels = [qr.name for qr in qc_u.qregs]

    dict_q_indices = get_qubit_indices(qc)

    indices_to_compose = [dict_q_indices[label] for label in qc_u_qregs_labels]

    # flatten the list of lists above
    indices_to_compose = [idx for sublist in indices_to_compose for idx in sublist]

    qc = qc.compose(qc_u, qubits=indices_to_compose)
    
    ###################################################
    
    if draw:
        
        qc.barrier()
            
        print("\nStep 3:")
        show_figure(qc.draw("mpl"))
        
    return qc

#######################################################################################
#######################################################################################

def step4(qc, draw=False):
    
    qc.h(-1)

    if draw:
        
        qc.barrier()
    
        print("\nStep 4:")
        show_figure(qc.draw("mpl"))
        
    return qc

#######################################################################################
#######################################################################################

def measurement(qc, draw_final=False):
    '''
    implements the measurement of the circuit
    '''
    
    # getting the registers
    registers = qc.qregs
    registers_labels = [qr.name for qr in qc.qregs]

    qr_anc = registers[registers_labels.index("anc")]
    qr_target = registers[registers_labels.index("target_train")]
    
    ###########################3
    
    cr_anc = ClassicalRegister(1, "c_anc")
    qc_cr_anc = QuantumCircuit(cr_anc)

    cr_class = ClassicalRegister(1, "c_class")
    qc_cr_class = QuantumCircuit(cr_class)

    # qc = qc + qc_cr_anc + qc_cr_class
    # new version: use tensor
    qc = qc_cr_class.tensor(qc_cr_anc).tensor(qc)

    qc.measure(qr_anc, cr_anc)
    qc.measure(qr_target, cr_class)

    if draw_final:
        print("\nStep 5 (measurement):")
        show_figure(qc.draw("mpl"))
        
    return qc

#######################################################################################
#######################################################################################

def run(qc, X_test, y_proba_kind="list", plot_stuff=False, print_stuff=False, print_simple=False):
    '''
    runs the circuit, display results and return both
    predicted class and probability distribution of prediction
    '''

    backend = Aer.get_backend('aer_simulator')
    try:
        backend.set_options(device='GPU')
    except:
        pass
        
    # backend = Aer.get_backend("qasm_simulator")
    
    n_shots = 1e4

    job = execute(qc, backend, shots=n_shots, seed_simulator=42)
    results = job.result()
    counts = results.get_counts()

    # filtering out measurements of ancila = |1> (last register)
    keys = list(counts.keys()).copy()
    for key in keys:

        if key[-1] == "1":

            counts.pop(key)

    if plot_stuff:
        show_figure(plot_histogram(counts, title="Results", figsize=(12, 4)))

    ####################################
    # final prediction
    
    # dictionary of counts sorted by value (sbv), decreasing order...
    counts_sbv = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    # ...will have its first key (first [0]) corresponding to the class with highest count (thus, probability!)
    # the key is of the form "class ancilla", thus, to get the actual class, get the first character (second [0])
    # new: predicted class is transformed to integer
    y_pred = int(list(counts_sbv.keys())[0][0])

    # also display probabilities (predict_proba)
    n_norm = sum(list(counts.values()))
    
    if y_proba_kind == "list":
        
        # this is to MAKE SURE that y_proba is [prob(y=0), prob(y=1)]
        counts_aux = {int(list(k)[0][0]) : v for k, v in counts.items()}
        
        # this is very important, in the case that there's only a single count
        if len(counts_aux) == 1:
    
            if list(counts_aux.keys())[0] == 0:

                counts_aux[1] = 0

            else:

                counts_aux[0] = 0

        y_proba = [counts_aux[0]/n_norm, counts_aux[1]/n_norm]
        
    elif y_proba_kind == "dict":
        
        y_proba = {f"p(c={classe[0]})" : count/n_norm for classe, count in counts.items()}
        
    else:
        
        raise ValueError("invalid y_proba_kind!")
        
    if print_stuff:
        
        print("\nProbability of belonging to each class:\n")

        if y_proba_kind == "dict":
            
            for k, v in y_proba.items():

                print("{} = {:.3f}".format(k, v))
        
        else:
            
            print("p(c=0) = {:.3f}".format(y_proba[0]))
            print("p(c=1) = {:.3f}".format(y_proba[1]))
                
        print("\n")
        print("*"*30)
        print("="*30)
        print("*"*30)
        print("\n")

        print("Final prediction:\n")
        print(f"y({X_test}) = {y_pred}")
        
    elif print_simple:
        
        print(f"\nObservation {X_test} classified!")
    
    return y_pred, y_proba

#######################################################################################
#######################################################################################

def quantum_nearest_neighbors(qc, n, X_test, draw=False, draw_final=False, 
                              y_proba_kind="list", plot_stuff=False, 
                              print_stuff=False, print_simple=False, 
                              run_flag=True):
    '''
    final wrapper function.
    optionally, runs the circuit or only returns the qc (argument `run`)
    this behavior is useful for the `construct_circuit` method!
    '''
    
    qc = step1(qc, draw)
    
    qc = step2(qc, n, draw)
    
    qc = step3(qc, n, draw)
    
    qc = step4(qc, draw)
    
    qc = measurement(qc, draw_final)
    
    if run_flag:
        
        y_pred, y_proba = run(qc, X_test, y_proba_kind, plot_stuff, print_stuff, print_simple)
    
        return y_pred, y_proba
    
    else:
        
        return qc

#######################################################################################
#######################################################################################

def drop_duplicates_and_correct_ambiguities(df_bin_dd, how):
    '''
    function used to the desambiguation of the target after binarization.
    - how: "major", "minor", "first", strategy of sesambiguation. 
            None if no desambuiguation is to be performed.
    '''
    
    # in the case there's some desambiguation
    if how:
    
        if how not in ["major", "minor", "first"]:
            assert ValueError("Invalid desambiguation strategy (parameter 'how')!")

        # unique observations -- that's, effectively, the training dataset
        print(f"At first, we have {df_bin_dd.shape[0]} unique observations, below:")
        display(df_bin_dd)

        if how == "minor":
            # after I drop duplicates, what's the minority class?
            # this is important because it will be used for the ambiguous cases,
            # ir order to have a better balance of the classes
            desamb_with = df_bin_dd["y"].value_counts().index[-1]

        elif how == "major":
            # in this case, I keep the majority class, it's a possible option!
            desamb_with = df_bin_dd["y"].value_counts().index[0]

        else:
            # in this case, I only take the first thing that appears
            desamb_with = df_bin_dd["y"].iloc[0]

        # after dropping duplicates, which observations have same features, but different target?
        ambiguities = df_bin_dd[df_bin_dd.duplicated(subset=df_bin_dd.columns[:-1], keep=False)]

        print("\nAmbiguous observations (same features, different target):")
        display(ambiguities)

        # this is important for the case that there's more than one ambiguity
        for idxs_group in ambiguities.groupby(ambiguities.columns[:-1].tolist()).groups.values():

            # fill ambiguity with "desamb_with"
            ambiguities.loc[idxs_group, "y"] = desamb_with

        # now, I set these new desambiguized target to the dataset with duplicates drapped
        df_bin_dd.loc[ambiguities.index, "y"] = ambiguities["y"]

        # and now I drop duplicates considering only teh features!
        df_bin_dd = df_bin_dd.drop_duplicates(keep='first', subset=df_bin_dd.columns[:-1])

        print(f"\nActual observations, after target desambiguation:\n")
        display(df_bin_dd)
        
    # if how=None, no desambiguation
    else:
        
        print("\nNo desambiguation will be performed!\n")

    return df_bin_dd

#######################################################################################
#######################################################################################

class QNN():
    
    def __init__(self):
        
        self.model = "quantum nearest neighbors"
        
    def fit(self, X_train, y_train, how):
        '''
        data must be in array format!!
        '''
        
        # eliminate duplicates!!
        aux_train = (pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name="y")], axis=1)
                       .drop_duplicates(keep="first"))
        
        aux_train = drop_duplicates_and_correct_ambiguities(aux_train, how)
        
        X_train_ = aux_train.drop(columns="y")
        y_train_ = aux_train["y"]
        
        self.X_train = X_train_.to_numpy()
        
        self.y_train = y_train_.to_numpy()
        
        return self

    def check_dimensions(self, X_test):
        '''
        consistency check of X_test dimensions, and padding if necessary
        '''
        # if x_test is a dataframe, transform it to numpy array.
        self.X_test = X_test.to_numpy() if isinstance(X_test, pd.core.frame.DataFrame) else X_test
    
        # NEW 17/10/2021: important consistency check - if number of features is the same between train and test!
        # it given the preprocessing, it could happen that they would differ.
        # If test dims are less than train dims, we just pad with zeros
        # otherwise, we raise an error.
        
        # check for string input
        if isinstance(self.X_test, str):
            
            train_dims = self.X_train.shape[1]
            test_dims = len(self.X_test)
            
            # in this case, i'll just try some padding
            if test_dims < train_dims:
                
                self.X_test = self.X_test + "0"*(train_dims - test_dims)
            
            # in this case, there's not much to be done...
            # but i think this won't happen. let's cover it anyway, though
            elif test_dims > train_dims:
                
                error_str = "Dimensionality mismatch!"
                error_str += f"\nTraining data has {self.X_train.shape[1]} dimensions"
                error_str += f"\nAnd test dat has {self.X_test.shape[1]} dimensions!"
                error_str += f"\nPlease check pre-processing!"
                
                raise ValueError(error_str)
         
        # check for numpy (dataframe was already converted in the first step, above
        else:
            
            train_dims = self.X_train.shape[1]
            test_dims = self.X_test.shape[1]
            
            # in this case, i'll just try some padding
            if test_dims < train_dims:
                
                self.X_test = np.pad(self.X_test, 
                                     [(0, 0), (0, train_dims - test_dims)],
                                     mode='constant', 
                                     constant_values=0).copy()
            
            # in this case, there's not much to be done...
            # but i think this won't happen. let's cover it anyway, though
            elif test_dims > train_dims:
                
                error_str = "Dimensionality mismatch!"
                error_str += f"\nTraining data has {self.X_train.shape[1]} dimensions"
                error_str += f"\nAnd test dat has {self.X_test.shape[1]} dimensions!"
                error_str += f"\nPlease check pre-processing!"
                
                raise ValueError(error_str)
        
    
    def predict(self, X_test, y_proba_kind="list", plot_stuff=False, print_stuff=False, print_simple=False):
        '''
        returns both the target class (y_pred) as well as 
        the probabilities of belonging to each class (y_proba).
        
        - y_proba_kind: "list" or "dict", dictates how the probabilities are returned.
            if "list": [p0, p1]
            if "dict": {"p(c=0)" : p0, "p(c=1)" : p1} 
        '''
       
        # X_test dimensions consistency check
        self.check_dimensions(X_test)
        
        # cache, very important
        self.X_test_cache = {}
        
        # n_repeat = 1 if there's a single observation to predict, in the form of a string
        n_repeat = self.X_test.shape[0] if isinstance(self.X_test, (np.ndarray, pd.core.frame.DataFrame)) else 1
        
        y_pred = []
        y_proba = []
        
        for i in range(n_repeat):

            if isinstance(self.X_test, (np.ndarray, pd.core.frame.DataFrame)):
                
                X_test_single = "".join(self.X_test[i, :].astype(str).tolist())
               
            # only perform quantum computation if observation is new
            # (that is, if after performing biarization, the resulting row (features values) weren't yet been processed)
            if X_test_single not in self.X_test_cache:
                
                qc, n = construct_initial_state(self.X_train, self.y_train, X_test_single)

                y_pred_single, y_proba_single = quantum_nearest_neighbors(qc, n, X_test_single, 
                                                                          y_proba_kind=y_proba_kind, 
                                                                          plot_stuff=plot_stuff, 
                                                                          print_stuff=print_stuff,
                                                                          print_simple=print_simple)
                
                self.X_test_cache[X_test_single] = (y_pred_single, y_proba_single)
               
            # if test observation is repeated, just get predictions from cache
            else:
                
                y_pred_single, y_proba_single = self.X_test_cache[X_test_single]
                
                if print_simple:
                    
                    print(f"\nObservation {X_test_single} classified (using cache)!")
                
            
            y_pred.append(y_pred_single)
            
            y_proba.append(y_proba_single)
        
        return y_pred, y_proba
    
    
    def construct_circuit(self, X_test, k=0, y_proba_kind="list", plot_stuff=False, print_stuff=False, print_simple=False):
        '''
        this only construct and returns the quantum circuit for a given test observation (param "k").
        it's usefull to study the quantum circuit.
        '''
        
        self.check_dimensions(X_test)
           
        n_repeat = 1
        
        if isinstance(self.X_test, (np.ndarray, pd.core.frame.DataFrame)):

            # only the k-th observation
            X_test_single = "".join(self.X_test[k, :].astype(str).tolist())

        qc, n = construct_initial_state(self.X_train, self.y_train, X_test_single)

        qc = quantum_nearest_neighbors(qc, n, X_test_single, 
                                       y_proba_kind=y_proba_kind, 
                                       plot_stuff=plot_stuff, 
                                       print_stuff=print_stuff,
                                       print_simple=print_simple,
                                       run_flag=False)
     
        return qc