/** 
* SVP inference C++ core
* 
* Author: Thomas Mortier
* Date: November 2021
*
* TODO: 
*   - check if we can add svp params as argument to init (avoids a couple of potential issues)
*   - clean code
*   - documentation
*   - comments
*   - improve runtime -> parallel processing of batch
*   - improve mem consumption
*/
#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <queue>
#include <math.h>
#include <tuple>
#include <sstream>
#include <string>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <random>
#include "svp_cpp.h"

void HNode::addch(int64_t in_features, std::vector<int64_t> y) {
    // check if leaf or internal node 
    if (this->chn.size() > 0)
    {
        // check if y is a subset of one of the children
        int64_t ind = -1;
        for (int64_t i=0; i<static_cast<int64_t>(this->chn.size()); ++i)
        {
            if (std::includes(this->chn[i]->y.begin(), this->chn[i]->y.end(), y.begin(), y.end()) == 1)
            {
                ind = i;
                break;
            }
        }
        if (ind != -1)
            // subset found, hence, recursively pass to child
            this->chn[ind]->addch(in_features, y);
        else
        {
            // no children for which y is a subset, hence, put in children list
            HNode* new_node = new HNode();
            new_node->y = y;
            new_node->chn = {};
            new_node->par = this->par;
            this->chn.push_back(new_node);
            unsigned long tot_len_y_chn {0};
            for (auto c : this->chn)
                tot_len_y_chn += c->y.size();
            // check if the current node has all its children
            if (tot_len_y_chn == this->y.size())
            {
                // get string representation of y
                std::stringstream ystr;
                std::copy(y.begin(), y.end(), std::ostream_iterator<int>(ystr, " "));
                this->estimator = this->par->register_module(ystr.str(), torch::nn::Linear(in_features, this->chn.size()));
            }
        }
    }
    else
    { 
        // no children yet, hence, put in children list
        HNode* new_node = new HNode();
        new_node->y = y;
        new_node->chn = {};
        new_node->par = this->par;
        this->chn.push_back(new_node);
    }
}

torch::Tensor HNode::forward(torch::Tensor input, torch::nn::CrossEntropyLoss criterion, int64_t y_ind) {
    torch::Tensor loss = torch::tensor({0}).to(input.device());
    torch::Tensor y = torch::tensor({y_ind}).to(input.device());
    if (this->chn.size() > 1)
    {
        auto o = this->estimator->forward(input);
        loss = loss + criterion(o, y);
    }

    return loss;
}

SVP::SVP(int64_t in_features, int64_t num_classes, std::vector<std::vector<int64_t>> hstruct) {
    this->num_classes = num_classes;
    // create root node 
    this->root = new HNode();
    if (hstruct.size() == 0) {
        this->root->estimator = this->register_module("linear", torch::nn::Linear(in_features,this->num_classes));
        this->root->y = {};
        this->root->chn = {};
        this->root->par = this;
    } else {
        // construct tree for h-softmax
        this->root->y = hstruct[0];
        this->root->chn = {};
        this->root->par = this;
        for (int64_t i=1; i<static_cast<int64_t>(hstruct.size()); ++i)
            this->root->addch(in_features, hstruct[i]);   
    }
}

SVP::SVP(int64_t in_features, int64_t num_classes, torch::Tensor hstruct) {
    this->num_classes = num_classes;
    this->hstruct = hstruct;
    // create root node 
    this->root = new HNode();
    this->root->estimator = this->register_module("linear", torch::nn::Linear(in_features,this->num_classes));
    this->root->y = {};
    this->root->chn = {};
    this->root->par = this;
}

torch::Tensor SVP::forward(torch::Tensor input, std::vector<std::vector<int64_t>> target) {
    torch::Tensor loss = torch::tensor({0}).to(input.device());
    torch::nn::CrossEntropyLoss criterion;
    // run over each sample in batch
    for (int64_t bi=0;bi<input.size(0);++bi)
    {
        // begin at root
        HNode* visit_node = this->root;
        for (int64_t yi=0;yi<static_cast<unsigned int>(target[bi].size());++yi)
        {
            auto o = visit_node->forward(input[bi].view({1,-1}), criterion, target[bi][yi]);
            loss = loss + o;
            visit_node = visit_node->chn[target[bi][yi]];
        }
    }
    loss = loss/input.size(0);
        
    return loss;
}

torch::Tensor SVP::forward(torch::Tensor input, torch::Tensor target) {
    torch::Tensor loss = torch::tensor({0});
    torch::nn::CrossEntropyLoss criterion;
    auto o = this->root->estimator->forward(input);
    loss = criterion(o, target);

    return loss;
}

std::vector<int64_t> SVP::predict(torch::Tensor input) {
    if (this->root->y.size() == 0)
    {
        auto o = this->root->estimator->forward(input);
        o = o.argmax(1).to(torch::kInt64);
        o = o.to(torch::kCPU);
        std::vector<int64_t> prediction(o.data_ptr<int64_t>(), o.data_ptr<int64_t>() + o.numel());

        return prediction;
    }
    else
    {
        std::vector<int64_t> prediction;
        // run over each sample in batch
        for (int64_t bi=0;bi<input.size(0);++bi)
        {
            // begin at root
            HNode* visit_node = this->root;
            while (visit_node->y.size() > 1)
            {
                auto o = visit_node->estimator->forward(input[bi].view({1,-1}));
                int64_t max_ch_ind = o.argmax(1).item<int64_t>();
                visit_node = visit_node->chn[max_ch_ind];
            }
            prediction.push_back(visit_node->y[0]);
        }
        return prediction;
    }
}

std::vector<std::vector<int64_t>> SVP::predict_set_fb(torch::Tensor input, int64_t beta, int64_t c) {
    std::vector<std::vector<int64_t>> prediction;
    // init problem
    param p;
    p.constr = ConstraintType::NONE;
    p.beta = beta;
    p.c = c;
    prediction = this->predict_set(input, p);
    
    return prediction;
}

std::vector<std::vector<int64_t>> SVP::predict_set_size(torch::Tensor input, int64_t size, int64_t c) {
    std::vector<std::vector<int64_t>> prediction;
    // init problem
    param p;
    p.constr = ConstraintType::SIZE;
    p.size = size;
    p.c = c;
    prediction = this->predict_set(input, p);
    
    return prediction;
}

std::vector<std::vector<int64_t>> SVP::predict_set_error(torch::Tensor input, double error, int64_t c) {
    std::vector<std::vector<int64_t>> prediction;
    // init problem
    param p;
    p.constr = ConstraintType::ERROR;
    p.error = error;
    p.c = c;
    prediction = this->predict_set(input, p);
    
    return prediction;
}

std::vector<std::vector<int64_t>> SVP::predict_set(torch::Tensor input, const param& p) {
    std::vector<std::vector<int64_t>> prediction;
    if ((this->root->y.size() == 0) && (p.c == this->num_classes)) {
        prediction = this->gsvbop(input, p);
    } else if ((this->root->y.size() == 0) && (p.c < this->num_classes)) {
        prediction = this->gsvbop_r(input, p);
    } else if ((this->root->y.size() > 0) && (p.c == this->num_classes)) {
        prediction = this->gsvbop_hf(input, p);
    } else {
        prediction = this->gsvbop_hf_r(input, p);
    }
    
    return prediction;
}

std::vector<std::vector<int64_t>> SVP::gsvbop(torch::Tensor input, const param& p) {
    std::vector<std::vector<int64_t>> prediction;
    auto o = this->root->estimator->forward(input);
    o = torch::nn::functional::softmax(o, torch::nn::functional::SoftmaxFuncOptions(1));
    // sort probabilities in decreasing order
    torch::Tensor idx {torch::argsort(o, 1, true).to(torch::kCPU)};
    // run over each sample in batch
    for (int64_t bi=0; bi<input.size(0); ++bi)
    {
        std::vector<int64_t> yhat;
        double yhat_p {0.0};
        std::vector<int64_t> ystar;
        double ystar_u {0.0};
        for (int64_t yi=0; yi<idx.size(1); ++yi)
        {
            yhat.push_back(idx[bi][yi].item<int64_t>());
            yhat_p += o[bi][idx[bi][yi]].to(torch::kCPU).item<double>();
            if (p.constr == ConstraintType::NONE) {
                double yhat_u {yhat_p*(1.0+pow(p.beta,2.0))/(yi+1+pow(p.beta,2.0))};
                if (yhat_u >= ystar_u) {
                    ystar = yhat;
                    ystar_u = yhat_u;
                } else {
                    break;
                }
            } else if (p.constr == ConstraintType::SIZE) {
                if (yi+1 > p.size) {
                    break;
                } else {
                    if (yhat_p >= ystar_u) {
                        ystar = yhat;
                        ystar_u = yhat_p;
                    }
                }
            } else {
                if (yhat_p >= 1-p.error) {
                    ystar = yhat;
                    break;
                }
            }
        }
        prediction.push_back(ystar);
    }

    return prediction;
}

std::vector<std::vector<int64_t>> SVP::gsvbop_r(torch::Tensor input, const param& p) {
    std::vector<std::vector<int64_t>> prediction;
    auto o = this->root->estimator->forward(input);
    o = torch::nn::functional::softmax(o, torch::nn::functional::SoftmaxFuncOptions(1));
    o = torch::matmul(o,this->hstruct.t().to(torch::kFloat32));
    // run over each sample in batch
    for (int64_t bi=0; bi<input.size(0); ++bi)
    {
        std::vector<int64_t> pred;
        int64_t si_optimal {0};
        double si_optimal_u {0.0};
        for (int64_t si=0; si<o.size(1); ++si)
        {
            double si_curr_p {o[bi][si].to(torch::kCPU).item<double>()};
            if (p.constr == ConstraintType::NONE) {
                double si_curr_size {this->hstruct.index({si,"..."}).sum().to(torch::kCPU).item<double>()};
                double si_curr_u {si_curr_p*(1.0+pow(p.beta,2.0))/(si_curr_size+pow(p.beta,2.0))};
                if (si_curr_u >= si_optimal_u) {
                    si_optimal = si;
                    si_optimal_u = si_curr_u;
                }
            } else if (p.constr == ConstraintType::SIZE) {
                if (si_curr_p >= si_optimal_u) {
                    si_optimal = si;
                    si_optimal_u = si_curr_p;
                }
            } else {
                if (si_curr_p >= 1.0-p.error) {
                    // also calculate set size
                    double si_curr_u {1.0/this->hstruct.index({si,"..."}).sum().to(torch::kCPU).item<double>()};
                    if (si_curr_u >= si_optimal_u) {
                        si_optimal = si;
                        si_optimal_u = si_curr_u;
                    }
                }
            }
        }
        // we have found the optimal solution, hence, update
        for (int64_t si=0; si<this->hstruct.size(1); ++si)
        {
            if (this->hstruct.index({si_optimal,si}).to(torch::kCPU).item<int>() == 1)
                pred.push_back(si);
        }
        prediction.push_back(pred);
    }

    return prediction;
}

std::vector<std::vector<int64_t>> SVP::gsvbop_hf(torch::Tensor input, const param& p) {
    std::vector<std::vector<int64_t>> prediction;
    // run over each sample in batch
    for (int64_t bi=0; bi<input.size(0); ++bi)
    {
        std::vector<int64_t> ystar;
        double ystar_u {0.0};
        std::vector<int64_t> yhat;
        double yhat_p {0.0};
        std::priority_queue<QNode> q;
        q.push({this->root, 1.0});
        while (!q.empty()) {
            QNode current {q.top()};
            q.pop();
            if (current.node->y.size() == 1) {
                // update solution
                yhat.push_back(current.node->y[0]);
                yhat_p += current.prob;
                if (p.constr == ConstraintType::NONE) {
                    double yhat_u {yhat_p*(1.0+pow(p.beta,2.0))/(yhat.size()+pow(p.beta,2.0))};
                    if (yhat_u >= ystar_u) {
                        ystar = yhat;
                        ystar_u = yhat_u;
                    } else {
                        break;
                    }
                } else if (p.constr == ConstraintType::SIZE) {
                    if (static_cast<int64_t>(yhat.size()) > p.size) {
                        break;
                    } else {
                        if (yhat_p >= ystar_u) {
                            ystar = yhat;
                            ystar_u = yhat_p;
                        }
                    }
                } else {
                    if (yhat_p >= 1-p.error) {
                        ystar = yhat;
                        break;
                    }
                }
            } else {
                // forward step
                auto o = current.node->estimator->forward(input[bi].view({1,-1}));
                o = torch::nn::functional::softmax(o, torch::nn::functional::SoftmaxFuncOptions(1)).to(torch::kCPU);
                for (int64_t i = 0; i<static_cast<int64_t>(current.node->chn.size()); ++i)
                {
                    HNode* c_node {current.node->chn[i]};
                    q.push({c_node, current.prob*o[0][i].item<double>()});
                }
            }
        }
        prediction.push_back(ystar);
    }

    return prediction;
}

std::vector<std::vector<int64_t>> SVP::gsvbop_hf_r(torch::Tensor input, const param& p) {
    std::vector<std::vector<int64_t>> prediction;
    // run over each sample in batch
    for (int64_t bi=0; bi<input.size(0); ++bi)
    {
        std::vector<int64_t> ystar;
        double ystar_u {0.0};
        std::vector<int64_t> yhat;
        double yhat_p {0.0};
        std::priority_queue<QNode> q;
        q.push({this->root, 1.0});
        std::tuple<std::vector<int64_t>, double> bop {this->_gsvbop_hf_r(input[bi].view({1,-1}), p, p.c, ystar, ystar_u, yhat, yhat_p, q)};
        prediction.push_back(std::get<0>(bop));
    }

    return prediction;
}

std::tuple<std::vector<int64_t>, double> SVP::_gsvbop_hf_r(torch::Tensor input, const param& p, int64_t c, std::vector<int64_t> ystar, double ystar_u, std::vector<int64_t> yhat, double yhat_p, std::priority_queue<QNode> q) {
    std::tuple<std::vector<int64_t>, double> prediction;
    while (!q.empty()) {
        std::vector<int64_t> ycur {yhat};
        double ycur_p {yhat_p};
        QNode current {q.top()};
        q.pop();
        for (int64_t i=0; i<static_cast<int64_t>(current.node->y.size()); ++i) {
            ycur.push_back(current.node->y[i]);
        }
        ycur_p = ycur_p + current.prob;
        if (p.constr == ConstraintType::NONE) {
            double ycur_u = ycur_p*(1.0+pow(p.beta,2.0))/(static_cast<double>(ycur.size())+pow(p.beta,2.0));
            if (ycur_u >= ystar_u) {
                ystar = ycur;
                ystar_u = ycur_u;
            }
        } else if (p.constr == ConstraintType::SIZE) {
            if (static_cast<int64_t>(ycur.size()) <= p.size) {
                double ycur_u = ycur_p;
                if (ycur_u >= ystar_u) {
                    ystar = ycur;
                    ystar_u = ycur_u;
                } 
            }
        } else {
            if (ycur_p >= 1.0-p.error) {
                double ycur_u = 1.0/static_cast<double>(ycur.size());
                if (ycur_u >= ystar_u) {
                    ystar = ycur;
                    ystar_u = ycur_u;
                } 
            }
        }
        if (p.constr == ConstraintType::NONE) {
            if (c > 1) {
                std::tuple<std::vector<int64_t>, double> bop {this->_gsvbop_hf_r(input, p, c-1, ystar, ystar_u, ycur, ycur_p, q)};
                ystar = std::get<0>(bop);
                ystar_u = std::get<1>(bop);
            }
        } else if (p.constr == ConstraintType::SIZE) {
            if (static_cast<int64_t>(ycur.size()) <= p.size) {
                if (c > 1) {
                    std::tuple<std::vector<int64_t>, double> bop {this->_gsvbop_hf_r(input, p, c-1, ystar, ystar_u, ycur, ycur_p, q)};
                    ystar = std::get<0>(bop);
                    ystar_u = std::get<1>(bop);
                } else {
                    break;
                }
            }
        } else if (p.constr == ConstraintType::ERROR) {
            if (ycur_p < 1.0-p.error) {
                if (c > 1) {
                    std::tuple<std::vector<int64_t>, double> bop {this->_gsvbop_hf_r(input, p, c-1, ystar, ystar_u, ycur, ycur_p, q)};
                    ystar = std::get<0>(bop);
                    ystar_u = std::get<1>(bop);
                } else {
                    break;
                }
            }
        }
        if (current.node->y.size() > 1) {
            // forward step
            auto o = current.node->estimator->forward(input);
            o = torch::nn::functional::softmax(o, torch::nn::functional::SoftmaxFuncOptions(1)).to(torch::kCPU);
            for (int64_t i = 0; i<static_cast<int64_t>(current.node->chn.size()); ++i)
            {
                HNode* c_node {current.node->chn[i]};
                q.push({c_node, current.prob*o[0][i].item<double>()});
            }
        } else {
            break;
        }
    }
    std::get<0>(prediction) = ystar;
    std::get<1>(prediction) = ystar_u;

    return prediction;
}
    
/* cpp->py bindings */ 
PYBIND11_MODULE(svp_cpp, m) {
    using namespace pybind11::literals;
    torch::python::bind_module<SVP>(m, "SVP")
        .def(py::init<int64_t, int64_t, std::vector<std::vector<int64_t>>>(), "in_features"_a, "num_classes"_a, "hstruct"_a=py::list())
        .def(py::init<int64_t, int64_t, torch::Tensor>(), "in_features"_a, "num_classes"_a, "hstruct"_a)
        .def("forward", py::overload_cast<torch::Tensor, torch::Tensor>(&SVP::forward))
        .def("forward", py::overload_cast<torch::Tensor, std::vector<std::vector<int64_t>>>(&SVP::forward))
        .def("predict", &SVP::predict, "input"_a)
        .def("predict_set_fb", &SVP::predict_set_fb, "input"_a, "beta"_a, "c"_a)
        .def("predict_set_size", &SVP::predict_set_size, "input"_a, "size"_a, "c"_a)
        .def("predict_set_error", &SVP::predict_set_error, "input"_a, "error"_a, "c"_a);
}
