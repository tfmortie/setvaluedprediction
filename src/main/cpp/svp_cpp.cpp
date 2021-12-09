/** 
* Implementation of SVP by using the LibTorch C++ frontend.
* 
* Author: Thomas Mortier
* Date: November 2021
*
*
* TODO: 
*   - documentation
*   - improve runtime -> parallel processing of batch
*/
#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <queue>
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
    torch::Tensor loss = torch::tensor({0});
    torch::Tensor y = torch::tensor({y_ind});
    if (this->chn.size() > 1)
    {
        auto o = this->estimator->forward(input);
        loss = loss + criterion(o, y);
    }
    return loss;
}

SVP::SVP(int64_t in_features, int64_t num_classes, std::vector<std::vector<int64_t>> hstruct) {
    // first create root node 
    this->root = new HNode();
    // check if we need a softmax or h-softmax
    if (hstruct.size() == 0) {
        this->root->estimator = this->register_module("linear", torch::nn::Linear(in_features,num_classes));
        this->root->y = {};
        this->root->chn = {};
        this->root->par = this;
    }
    else
    {
        // construct tree for h-softmax
        this->root->y = hstruct[0];
        this->root->chn = {};
        this->root->par = this;
        for (int64_t i=1; i<static_cast<int64_t>(hstruct.size()); ++i)
            this->root->addch(in_features, hstruct[i]);   
    }
}

torch::Tensor SVP::forward(torch::Tensor input, std::vector<std::vector<int64_t>> target) {
    torch::Tensor loss = torch::tensor({0});
    torch::nn::CrossEntropyLoss criterion;
    if (this->root->y.size() == 0)
    {
        auto o = this->root->estimator->forward(input);
        for (int64_t bi=0;bi<input.size(0);++bi)
            loss = loss + criterion(o[bi], torch::tensor({target[bi][0]}));
        return loss/input.size(0);
    }
    else
    {
        // run over each sample in batch
        for (int64_t bi=0;bi<input.size(0);++bi)
        {
            // begin at root
            HNode* visit_node = this->root;
            for (int64_t yi=0;yi<static_cast<unsigned int>(target[bi].size());++yi)
            {
                loss = loss + visit_node->forward(input[bi].view({1,-1}), criterion, target[bi][yi]);
                visit_node = visit_node->chn[target[bi][yi]];
            }
        }
        return loss/input.size(0);
    }
}

PYBIND11_MODULE(svp_cpp, m) {
    using namespace pybind11::literals;
    torch::python::bind_module<SVP>(m, "SVP")
        .def(py::init<int64_t, int64_t, std::vector<std::vector<int64_t>>>(), "in_features"_a, "num_classes"_a, "hstruct"_a=py::list())
        .def("forward", &SVP::forward, "input"_a, "target"_a=py::list());
}
