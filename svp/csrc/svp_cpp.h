/** 
* Header SVP inference C++ core
* 
* Author: Thomas Mortier
* Date: November 2021
*
*/
#ifndef SVP_H
#define SVP_H

#include <torch/torch.h>

/* Set-valued predictor type */
enum class SVPType {
    FB,
    DG,
    SIZECTRL,
    ERRORCTRL,
    AVGERRORCTRL,
    APSAVGERRORCTRL
};

/* Defines the set-valued prediction problem */
struct param
{
    SVPType svptype {SVPType::FB}; /* by default (1/x)-utility maximization by means of f-measure */
    int64_t beta {1};
    double delta {1.6};
    double gamma {0.6};
    int64_t size {1};
    double error {0.05};
    int64_t c {1}; /* representation complexity */
};

/* Structure which represents the basic component for SVP  */
struct HNode : torch::nn::Module {
    // attributes
    torch::nn::Linear estimator {nullptr};
    std::vector<int64_t> y;
    std::vector<HNode*> chn;
    torch::nn::Module *par;
    HNode* parent;
    // functions
    void addch(int64_t in_features, std::vector<int64_t> y, int64_t id); 
    torch::Tensor forward(torch::Tensor input, torch::nn::CrossEntropyLoss criterion, int64_t y_ind={});
};

/* PQ struct used for inference */
struct QNode
{
    HNode* node;
    double prob;
    /* comparator */
    bool operator<(const QNode& n) const { return prob < n.prob;}
};

/* Class which represents an SVP object */
struct SVP : torch::nn::Module {
    // attributes
    int64_t num_classes;
    torch::Tensor hstruct;
    HNode* root;
    // forward-pass functions
    SVP(int64_t in_features, int64_t num_classes, std::vector<std::vector<int64_t>> hstruct={});
    SVP(int64_t in_features, int64_t num_classes, torch::Tensor hstruct);
    torch::Tensor forward(torch::Tensor input, std::vector<std::vector<int64_t>> target={}); /* forward pass for hierarchical model */
    torch::Tensor forward(torch::Tensor input, torch::Tensor target={}); /* forward pass for flat model */
    torch::Tensor forward(torch::Tensor input); /* forward pass for flat model */
    std::vector<int64_t> predict(torch::Tensor input); /* top-1 prediction */
    // set-valued prediction functions
    std::vector<std::vector<int64_t>> predict_set_fb(torch::Tensor input, int64_t beta, int64_t c);
    std::vector<std::vector<int64_t>> predict_set_dg(torch::Tensor input, double delta, double gamma, int64_t c);
    std::vector<std::vector<int64_t>> predict_set_size(torch::Tensor input, int64_t size, int64_t c);
    std::vector<std::vector<int64_t>> predict_set_error(torch::Tensor input, double error, int64_t c);
    std::vector<std::vector<int64_t>> predict_set_avgerror(torch::Tensor input, double error, int64_t c);
    std::vector<std::vector<int64_t>> predict_set(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> crsvphf(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> cusvphf(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> csvp(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> acrsvp(torch::Tensor input, torch::Tensor u, torch::Tensor u, const param& params);
    std::vector<std::vector<int64_t>> acusvp(torch::Tensor input, torch::Tensor u, const param& params);
    std::vector<std::vector<int64_t>> acsvp(torch::Tensor input, torch::Tensor u, const param& params);
    std::vector<std::vector<int64_t>> gsvbop(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> gsvbop_r(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> gsvbop_hf(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> gsvbop_hf_r(torch::Tensor input, const param& params);
    std::tuple<std::vector<int64_t>, double> _gsvbop_hf_r(torch::Tensor input, const param& params, int64_t c, std::vector<int64_t> ystar, double ystar_u, std::vector<int64_t> yhat, double yhat_p, std::priority_queue<QNode> q);
    // other functions
    void set_hstruct(torch::Tensor hstruct);
};

#endif