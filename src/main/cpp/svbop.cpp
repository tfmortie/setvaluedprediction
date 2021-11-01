/* 
    Author: Thomas Mortier 2021
    Implementation of SVBOP.
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
#include "svbop.h"

/* Code for SVPNode */

/*
  TODO
*/
void SVPNode::addch(int64_t in_features, std::vector<int64_t> y) {
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
      SVPNode* new_node = new SVPNode();
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
        this->estimator = this->par->register_module(ystr.str(), torch::nn::Linear(in_features,2));
      }
    }
  }
  else
  { 
    // no children yet, hence, put in children list
    SVPNode* new_node = new SVPNode();
    new_node->y = y;
    new_node->chn = {};
    new_node->par = this->par;
    this->chn.push_back(new_node);
  }
}
    
/*
  TODO
*/
torch::Tensor SVPNode::forward(torch::Tensor input, int64_t x_ind, int64_t y_ind) {
  torch::Tensor prob;
  if (this->chn.size() > 1)
  {
    prob = this->estimator->forward(input);
    prob = torch::nn::functional::softmax(prob, torch::nn::functional::SoftmaxFuncOptions(1));
    prob = prob[x_ind][y_ind];
  }
  return prob;
}

/* Code for SVPredictor */

/*
  TODO
*/
SVBOP::SVBOP(int64_t in_features, int64_t num_classes, std::vector<std::vector<int64_t>> hstruct) {
  // first create root node 
  this->root = new SVPNode();
  // now construct tree
  this->root->y = hstruct[0];
  this->root->chn = {};
  this->root->par = this;
  for (int64_t i=1; i<static_cast<int64_t>(hstruct.size()); ++i)
    this->root->addch(in_features, hstruct[i]);   
}

/*
  TODO
*/
torch::Tensor SVBOP::forward(torch::Tensor input, std::vector<int64_t> target) {
  int64_t batch_size {input.size(0)};
  std::vector<torch::Tensor> probs;
  // run over each sample in batch
  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
  for (int64_t bi=start; bi<end; bi++)
  //for (int64_t bi=0;bi<batch_size;++bi)
  {
    // begin at root
    SVPNode* visit_node = this->root;
    torch::Tensor prob = torch::ones(1);
    while (!visit_node->chn.empty())
    {
        int64_t found_ind {-1};
        for (int64_t i=0; i<static_cast<int64_t>(visit_node->chn.size()); ++i)
        { 
            if (std::count(visit_node->chn[i]->y.begin(), visit_node->chn[i]->y.end(), target[bi]))
            {
                found_ind = i;
                break;
            }  
        }
        if (found_ind != -1)
        {
            prob = prob*visit_node->forward(input, bi, found_ind);
            visit_node = visit_node->chn[found_ind];
        }
    }
    probs.push_back(prob);
  }
  });
  return torch::stack(probs);
}

PYBIND11_MODULE(svbop_cpp, m) {
   torch::python::bind_module<SVBOP>(m, "SVBOP")
     .def(py::init<int64_t, int64_t, std::vector<std::vector<int64_t>>>())
     .def("forward", &SVBOP::forward);
}