/** 
* Header SVP
* 
* Author: Thomas Mortier
* Date: November 2021
*
*/
#ifndef SVP_H
#define SVP_H

#include <torch/torch.h>

/* structure which represents the basic component for SVP  */
struct HNode : torch::nn::Module {
    // attributes
    torch::nn::Linear estimator {nullptr};
    std::vector<int64_t> y;
    std::vector<HNode*> chn;
    torch::nn::Module *par;
    // functions
    void addch(int64_t in_features, std::vector<int64_t> y); 
    torch::Tensor forward(torch::Tensor input, torch::nn::CrossEntropyLoss criterion, int64_t y_ind={});
};

/* class which represents an SVP object */
struct SVP : torch::nn::Module {
    // attributes
    HNode* root;
    // functions
    SVP(int64_t in_features, int64_t num_classes, std::vector<std::vector<int64_t>> hstruct={});
    torch::Tensor forward(torch::Tensor input, std::vector<std::vector<int64_t>> target={});
};

#endif
