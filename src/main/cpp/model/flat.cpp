/* 
    Author: Thomas Mortier 2019-2020

    Implementation of model with softmax
*/

#include <sstream>
#include <iomanip>
#include <random>
#include <fstream>
#include <queue>
#include <ctime> 
#include <iterator>
#include <algorithm>
#include <functional>
#include <queue>
#include <limits>
#include "model/model.h"
#include "model/flat.h"
#include "data.h"
#include "model/mmath.h"
#include "Eigen/Dense"
#include "Eigen/SparseCore"

/* CONSTRUCTOR AND DESTRUCTOR */

/* constructor (training mode) */
FlatModel::FlatModel(problem* prob) : Model(prob)
{
    // create W, D, M and V matrix (M and V for Adam)
    this->W = Eigen::MatrixXd::Zero(prob->d, prob->hstruct[0].size());
    this->D = Eigen::MatrixXd::Zero(prob->d, prob->hstruct[0].size());
    this->M = Eigen::MatrixXd::Zero(prob->d, prob->hstruct[0].size());
    this->V = Eigen::MatrixXd::Zero(prob->d, prob->hstruct[0].size());

    // initialize W matrix
    inituw(this->W, static_cast<double>(-1.0/this->W.rows()), static_cast<double>(1.0/this->W.rows()));
}

/* constructor (predict mode) */
FlatModel::FlatModel(const char* model_file_name, problem* prob) : Model(model_file_name, prob)
{
    this->load(model_file_name);
}

/* PRIVATE */

/*
    Forward pass and backprop call.

    Arguments:
        x: sparse feature vector
        y: class for which to calculate the probability (needed for loss)
        t: time step in case of Adam optimization
    Return: 
        probability for class y (needed for loss)
*/
double FlatModel::update(const Eigen::SparseVector<double>& x, const unsigned long y, const unsigned long t)
{
    // forward step (Wtx)
    Eigen::VectorXd o = this->W.transpose() * x;
    // apply softmax
    softmax(o);
    // set derivatives 
    dvscalm(D, o, y-1, x);
    // and update in case we have processed a mini-batch
    if (t % this->prob->batchsize == 0)
    {
        // calculate the average gradients
        this->D = this->D/this->prob->batchsize;
        if (this->prob->optim == OptimType::SGD)
            sgd(this->W, this->D, this->prob->lr);
        else
            adam(this->W, this->D, this->M, this->V, this->prob->lr, std::floor(t/this->prob->batchsize));
        this->D.setZero();
    } 
    return o[y-1];
}

/* returns weights in string representation (row-major order, space separated) */
std::string FlatModel::getWeightVector()
{
    std::string ret_arr;
    // process all elements in row-major order
    for (unsigned long i=0; i<this->W.rows(); ++i)
    {
        for (unsigned long j=0; j<this->W.cols(); ++j)
        {
            std::stringstream str_stream;
            str_stream << std::to_string(this->W(i,j));
            ret_arr += str_stream.str();
            if (i!=this->W.rows()-1 || j!=this->W.cols()-1)
                ret_arr += ' ';
        }
    }
    return ret_arr;
}

/* set weights in string representation (row-major order, space separated) */ 
void FlatModel::setWeightVector(std::string w_str)
{
    // convert string to input stream
    std::istringstream istr_stream {w_str};
    // weights are separated by ' ', hence, split accordingly
    std::vector<std::string> tokens {std::istream_iterator<std::string> {istr_stream}, std::istream_iterator<std::string>{}};
    // run over weights in row-major order and save to W
    for (unsigned long i=0; i<this->W.rows(); ++i)
    {
        for (unsigned long j=0; j<this->W.cols(); ++j)
            this->W(i,j) = std::stod(tokens[(i*this->W.cols())+j]);
    }
}

/* PUBLIC */ 

/* print structure of classification problem */
void FlatModel::printStruct()
{
    std::cout << vecToArr(this->prob->hstruct[0]) << '\n';
}

/* print some general information about model */
void FlatModel::printInfo(const bool verbose)
{
    std::cout << "---------------------------------------------------\n";
    std::cout << "[info] Flat model: \n";
    std::cout << "---------------------------------------------------\n";
    std::cout << "  * Number of features              = " << this->W.rows() << '\n';
    std::cout << "  * Number of classes               = " << this->W.cols() << '\n';   
    if (verbose)
    {
        std::cout << "  * Structure =\n";
        this->printStruct();
    }
    std::cout << "---------------------------------------------------\n\n";
}

/* k-fold cross-validation */
void FlatModel::performCrossValidation(unsigned int k)
{
    std::cout << "---- " << k << "-Fold CV ----\n";
    // first create index vector
    std::vector<unsigned long> ind_arr;
    for(unsigned long i=0; i<this->prob->n; ++i)
        ind_arr.push_back(i);
    // now shuffle index vector 
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(ind_arr), std::end(ind_arr), rng);
    // calculate size of each test fold
    unsigned long ns_fold {this->prob->n/static_cast<unsigned long>(k)};
    // start kfcv
    unsigned int iter {0};
    while (iter < k)
    {
        std::cout << "FOLD " << iter+1 << '\n';
        // extract test fold indices
        std::vector<unsigned long>::const_iterator i_start = ind_arr.begin() + static_cast<long>(static_cast<unsigned long>(iter)*ns_fold);
        std::vector<unsigned long>::const_iterator i_stop = ind_arr.begin() + static_cast<long>(static_cast<unsigned long>((iter+1))*ns_fold);
        std::vector<unsigned long> testfold_ind(i_start, i_stop);
        // now start fitting 
        this->fit(testfold_ind, 0);
        // and validate on training and test fold
        double acc {0.0};
        double n_cntr {0.0};
        for (unsigned long n = 0; n<this->prob->n; ++n)
        {
            if (std::find(testfold_ind.begin(), testfold_ind.end(), n) == testfold_ind.end())
            {
                unsigned long pred {this->predict(this->prob->X[n])};
                unsigned long targ {this->prob->y[n]};
                acc += (pred==targ);
                n_cntr += 1.0;
            }
        }
        std::cout << "Training accuracy: " << (acc/n_cntr)*100.0 << "% \n";
        acc = 0.0;
        n_cntr = 0.0;
        for (unsigned long n = 0; n<testfold_ind.size(); ++n)
        {
            unsigned long pred {this->predict(this->prob->X[testfold_ind[n]])};
            unsigned long targ {this->prob->y[testfold_ind[n]]};
            acc += (pred==targ);
            n_cntr += 1.0;
        }
        std::cout << "Test accuracy: " << (acc/n_cntr)*100.0 << "% \n";
        ++iter;
    }
    // and finally reset again
    this->reset();
    std::cout << "-------------------\n\n";
}

/* reset model (ie, weights) */
void FlatModel::reset()
{
    // reinitialize W
    inituw(this->W, static_cast<double>(-1.0/this->W.rows()), static_cast<double>(1.0/this->W.rows()));
    // and D,M and V (M,V for Adam)
    this->D.setZero();
    this->M.setZero();
    this->V.setZero();
}

/* fit on data (in problem instance), while validating on instances with ind in ign_index (if applicable) */
void FlatModel::fit(const std::vector<unsigned long>& ign_index, const bool verbose)
{
    std::cout << "---------------------------------------------------------------------------------\n";
    if (ign_index.size() != 0)
        std::cout << "Fit model on train/val (" << this->prob->n-ign_index.size() << '/' << ign_index.size() << ") data ...\n";
    else
        std::cout << "Fit model on train (" << this->prob->n << ") data ...\n";
    std::cout << "---------------------------------------------------------------------------------\n";
    std::cout << "* #Features: " << this->prob->d << '\n';
    std::cout << "* #Classes: " << this->prob->hstruct[0].size() << '\n';
    std::cout << "* #Epochs: " << this->prob->ne << '\n';
    std::cout << "* Mini-batch size: " << this->prob->batchsize << '\n';
    std::cout << "* Patience: " << this->prob->patience << '\n';
    if (this->prob->optim == OptimType::SGD)
        std::cout << "* Optimizer: SGD\n";
    else
        std::cout << "* Optimizer: Adam\n";
    std::cout << "* Learning rate: " << this->prob->lr << '\n';
    std::cout << "---------------------------------------------------------------------------------\n";
    unsigned int e_cntr {0};
    int patience_counter {0};
    double prev_best_loss {std::numeric_limits<double>::max()};
    unsigned long t {0}; // in case of Adam optimization
    // reset W,D,M,V
    this->reset();
    while (e_cntr < this->prob->ne)
    {
        // init. train/holdout loss and counter
        double e_loss_train {0.0};
        double e_loss_holdout {0.0};
        double n_cntr_train {0.0};
        double n_cntr_holdout {0.0};
        // run over each instance in the training set
        for (unsigned long n = 0; n<this->prob->n; ++n)
        {
            if (std::find(ign_index.begin(), ign_index.end(), n) == ign_index.end())
            {
                t += 1; 
                Eigen::SparseVector<double> x {this->prob->X[n]};
                unsigned long y {this->prob->y[n]}; // our class 
                double i_p {this->update(x, y, t)};
                double i_loss {std::log2((i_p<=EPS ? EPS : i_p))};
                e_loss_train += -i_loss;
                n_cntr_train += 1;
            }
            else
            {
                // holdout example, hence, only compute loss
                Eigen::SparseVector<double> x {this->prob->X[n]};
                std::vector<unsigned long> yv{this->prob->y[n]}; // our class
                double i_p{this->predict_proba(x,yv)[0]};
                double i_loss {std::log2(i_p<=EPS ? EPS : i_p)};
                e_loss_holdout += -i_loss;
                n_cntr_holdout += 1;
            }
        }
        // average training loss
        e_loss_train = e_loss_train/n_cntr_train;
        if (ign_index.size() != 0)
        {
            // also calculate average holdout loss
            e_loss_holdout = e_loss_holdout/n_cntr_holdout;
            // check if we have an improvement, compared to previous epoch loss
            if (e_loss_holdout < (prev_best_loss-0.01))
            {
                prev_best_loss = e_loss_holdout;
                patience_counter = 0; // reset patience counter
            }
            else
            {
                // increase patience counter
                patience_counter += 1;
            }
            // check if we can early stop
            if (patience_counter == this->prob->patience)
            {
                std::cout << "Eearly stopping at epoch " << (e_cntr+1) << ": train loss " << e_loss_train << "  -  val loss  " << e_loss_holdout << '\n';
                break;
            }
            else
            {
                if (verbose)
                    std::cout << "Epoch " << (e_cntr+1) << ": train loss "<< e_loss_train << "  -  val loss  " << e_loss_holdout << '\n';
            }
        }
        else
        {
            if (verbose)
                std::cout << "Epoch " << (e_cntr+1) << ": loss "<< e_loss_train << '\n';
        }
        ++e_cntr;
    }
    if (verbose)
        std::cout << '\n';
}

/*
    Return class with highest probability mass.

    Arguments:
        x: sparse feature vector
    Return: 
        class
*/
unsigned long FlatModel::predict(const Eigen::SparseVector<double>& x)
{
    // forward step (Wtx)
    Eigen::VectorXd o = this->W.transpose() * x;
    // apply softmax
    softmax(o);
    Eigen::VectorXd::Index max_i;
    o.maxCoeff(&max_i);
    return max_i+1;
}

/*
    Calculate probability masses for one or more classes.

    Arguments:
        x: sparse feature vector
        yv: vector of classes for which to calculate probability mass
        p: vector of probabilities
    Return: 
        vector of probabilities
*/
std::vector<double> FlatModel::predict_proba(const Eigen::SparseVector<double>& x, const std::vector<unsigned long> yv)
{
    std::vector<double> probs; 
    // run over all labels for which we need to calculate probs
    // forward step (Wtx)
    Eigen::VectorXd o = this->W.transpose() * x;
    // apply softmax
    softmax(o);
    for (unsigned long y : yv)
        probs.push_back(o[y-1]);
    return probs;
}

/* 
    Implementation for unrestricted bayes-optimal predictor.
    See https://arxiv.org/abs/1906.08129. 
*/
std::vector<unsigned long> FlatModel::predict_ubop(const Eigen::SparseVector<double>& x)
{
    // initalize prediction set, with probability and expected utility
    std::vector<unsigned long> set;
    double set_prob {0.0};
    double set_eu {0.0};
    // initialize priority queue that sorts in decreasing order of probability
    std::priority_queue<std::pair<double,unsigned long>, std::vector<std::pair<double,unsigned long>>, std::less<std::pair<double,unsigned long>>> q;
    // forward step (Wtx)
    Eigen::VectorXd o = this->W.transpose() * x;
    // apply softmax
    softmax(o);
    // push probs to priority queue
    for (unsigned long i=0; i<o.size(); ++i)
        q.push(std::make_pair(o(i),i+1));
    // now run over tuples in priority queue 
    while (!q.empty())
    {
        // get current (prob, class)
        std::pair<double, unsigned long> current = q.top(); 
        // push class to prediction set and add probability
        set.push_back(current.second);
        set_prob += current.first;
        // compute utility according to Eq. (5)
        double current_eu {set_prob*g(set, this->prob->utility)};
        // check if current solution is worse than best solution so far (early stopping criterion)
        if (current_eu < set_eu)
        {
            // remove last element from set (because previous one was optimal) and break
            set.pop_back();
            break;
        }
        else
        {
            // set new optimal expected utility and pop first element from priority queue
            set_eu = current_eu;
            q.pop();
        }
    }
    return set;
}

/* 
    Implementation for restricted bayes-optimal predictor.
    See https://arxiv.org/abs/1906.08129. 
*/
std::vector<unsigned long> FlatModel::predict_rbop(const Eigen::SparseVector<double>& x)
{
    // initalize optimal prediction set and corresponding expected utility 
    std::vector<unsigned long> opt_set;
    double opt_set_eu {0.0};
    // forward step (Wtx)
    Eigen::VectorXd o = this->W.transpose() * x;
    // apply softmax
    softmax(o);
    // now run over candidate solutions in struct
    for (std::vector<unsigned long> c_set : this->prob->hstruct)
    {
        // calculate probability mass 
        double cur_set_prob {0.0};
        for (unsigned long y : c_set)
            cur_set_prob += o[y-1];
        // and expected utility for candidate
        double cur_set_eu {cur_set_prob*g(c_set, this->prob->utility)};
        // set new optimal solution, in case we have an improvement
        if (cur_set_eu > opt_set_eu)
        {
            opt_set = c_set;
            opt_set_eu = cur_set_eu;
        }
    }
    return opt_set;
}

/* get number of classes */
unsigned long FlatModel::getNrClass()
{
    return this->W.cols();
}

/* get number of features (bias included) */ 
unsigned long FlatModel::getNrFeatures()
{
    return this->W.rows();
}

/* save model to file */
void FlatModel::save(const char* model_file_name)
{
    std::cout << "Saving model to " << model_file_name << "...\n";
    // create output file stream
    std::ofstream model_file;
    model_file.open(model_file_name, std::ofstream::trunc);
    // STRUCT
    model_file << "struct [";
    // process all except last element
    for (unsigned int i=0; i<this->prob->hstruct.size()-1; ++i)
        model_file << vecToArr(this->prob->hstruct[i]) << ',';
    // and now last element
    model_file << vecToArr(this->prob->hstruct[this->prob->hstruct.size()-1]) << "]\n";
    // #features
    model_file << "nr_feature " << this->prob->d << '\n';
    // bias
    model_file << "bias " << (this->prob->bias >= 0.0 ? 1.0 : -1.0) << '\n';
    // weights
    model_file << "w \n";
    model_file << this->getWeightVector() << '\n';       
    // close file
    model_file.close();
}

/* load model from file */
void FlatModel::load(const char* model_file_name)
{
    std::cout << "Loading model from " << model_file_name << "...\n";
    //1. create prob instance, based on information in file: h_struct, nr_feature, bias
    std::ifstream in {model_file_name};
    std::string line;
    bool w_mode {0};
    try
    {
        while (std::getline(in, line))
        {
            // get tokens for line (ie class and index:ftval)
            std::istringstream istr_stream {line};
            if (!w_mode)
            {
                // not yet in w mode, hence, get tokens based on ' ' 
                std::vector<std::string> tokens {std::istream_iterator<std::string> {istr_stream}, std::istream_iterator<std::string>{}};
                if (tokens[0] == "struct")
                    this->prob->hstruct = strToHierarchy(tokens[1]);
                else if(tokens[0] == "nr_feature")
                    this->prob->d = static_cast<unsigned long>(std::stoi(tokens[1]));
                else if(tokens[0] == "bias")
                    this->prob->bias = std::stod(tokens[1]);
                else
                {
                    // set w mode
                    w_mode = 1;
                }
            }
            else
            {
                // first create W, D, M and V matrix (M and V for Adam)
                this->W = Eigen::MatrixXd::Random(this->prob->d, this->prob->hstruct[0].size());
                this->D = Eigen::MatrixXd::Zero(this->prob->d, this->prob->hstruct[0].size());
                this->M = Eigen::MatrixXd::Zero(this->prob->d, this->prob->hstruct[0].size());
                this->V = Eigen::MatrixXd::Zero(this->prob->d, this->prob->hstruct[0].size());
                this->setWeightVector(line);         
            } 
        }
    }
    catch(std::ifstream::failure e)
    {
        std::cerr << "[error] Exception " << e.what() << " catched!\n";
        exit(1);
    }
}