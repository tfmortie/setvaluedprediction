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
    // forward-pass functions
    SVP(int64_t in_features, int64_t num_classes, std::vector<std::vector<int64_t>> hstruct={});
    torch::Tensor forward(torch::Tensor input, std::vector<std::vector<int64_t>> target={}); /* forward pass for hierarchical model */
    torch::Tensor forward(torch::Tensor input, torch::Tensor target={}); /* forward pass for flat model */
    // set-valued prediction functions
    std::vector<int64_t> predict(torch::Tensor input);
    //std::vector<int64_t> predict_set_fb(torch::Tensor input, int64_t beta, int64_t c);
    //std::vector<int64_t> predict_set_size(torch::Tensor input, int64_t size, int64_t c);
    //std::vector<int64_t> predict_set_error(torch::Tensor input, int64_t error, int64_t c);
};

#endif
