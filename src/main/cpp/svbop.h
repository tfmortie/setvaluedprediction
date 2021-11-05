/* 
    Author: Thomas Mortier 2021
    Header SVBOP.
*/

#ifndef SVBOP_H
#define SVBOP_H

#include <torch/torch.h>

/* structure which represents the basic component for SVBOP  */
struct SVPNode : torch::nn::Module {
    // attributes
    torch::nn::Linear estimator {nullptr};
    std::vector<int64_t> y;
    std::vector<SVPNode*> chn;
    torch::nn::Module *par;
    // functions
    void addch(int64_t in_features, std::vector<int64_t> y); 
    torch::Tensor forward(torch::Tensor input, int64_t y_ind={});
};

/* class which represents an SVBOP object */
struct SVBOP : torch::nn::Module {
    // attributes
    SVPNode* root;
    // functions
    SVBOP(int64_t in_features, int64_t num_classes, std::vector<std::vector<int64_t>> hstruct={});
    //torch::Tensor forward(torch::Tensor input, std::vector<int64_t> target={});
    torch::Tensor forward(torch::Tensor input, std::vector<std::vector<int64_t>> target={});
};

#endif
