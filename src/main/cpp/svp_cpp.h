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

/* constraint type */
enum class ConstraintType {
    NONE,
    SIZE,
    ERROR
};

/* defines the set-valued prediction problem */
struct param
{
    ConstraintType constr {ConstraintType::NONE};
    int64_t beta {1};
    int64_t size {1};
    double error {0.05};
    int64_t c {1}; /* representation complexity */
};

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
    std::vector<int64_t> predict(torch::Tensor input); /* top-1 prediction */
    // set-valued prediction functions
    std::vector<std::vector<int64_t>> predict_set_fb(torch::Tensor input, int64_t beta, int64_t c);
    std::tuple<std::vector<int64_t>, double> predict_set(torch::Tensor input, const param& params, int64_t c, std::vector<int64_t> ystar, double ystar_u, std::vector<int64_t> yhat, double yhat_p, std::priority_queue<QNode> q);
    //std::vector<std::vector<int64_t>> predict_set_size(torch::Tensor input, int64_t size, int64_t c);
    //std::vector<std::vector<int64_t>> predict_set_error(torch::Tensor input, int64_t error, int64_t c);
};

/* PQ struct used for inference */
struct QNode
{
    HNode* node;
    double prob;
    /* comparator */
    bool operator<(const QNode& n) const { return prob < n.prob;}
};

#endif
