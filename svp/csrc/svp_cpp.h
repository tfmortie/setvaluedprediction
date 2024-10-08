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
    LAC,
    RAPS, /* see Romano et al. (2020); Angelopoulos et al. (2022)*/
    CSVPHF,
    CRSVPHF
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
    /* params for RAPS method (see Angelopoulos et al. (2020))*/
    bool rand {false};
    double lambda {0.0};
    int64_t k {1};
    int64_t c {1}; /* representation complexity */
};

/* Structure which represents the basic component for SVP  */
struct HNode : torch::nn::Module {
    // attributes
    torch::nn::Sequential estimator {nullptr};
    std::vector<int64_t> y;
    std::vector<HNode*> chn;
    torch::nn::Module *par;
    HNode* parent {nullptr};
    // functions
    void addch(int64_t in_features, double dp, std::vector<int64_t> y, int64_t id); 
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
    double db;
    torch::Tensor hstruct;
    HNode* root;
    std::unordered_map<HNode*, std::pair<double, std::vector<std::pair<std::vector<int64_t>,double>>>> sr_map;
    // forward-pass functions
    SVP(int64_t in_features, int64_t num_classes, double dp, std::vector<std::vector<int64_t>> hstruct={});
    SVP(int64_t in_features, int64_t num_classes, double dp, torch::Tensor hstruct);
    torch::Tensor forward(torch::Tensor input, std::vector<std::vector<int64_t>> target={}); /* forward pass for hierarchical model */
    torch::Tensor forward(torch::Tensor input, torch::Tensor target={}); /* forward pass for flat model */
    torch::Tensor forward(torch::Tensor input); /* forward pass for flat model */
    std::vector<int64_t> predict(torch::Tensor input); /* top-1 prediction */
    // set-valued prediction functions
    std::vector<std::vector<int64_t>> predict_set_fb(torch::Tensor input, int64_t beta, int64_t c);
    std::vector<std::vector<int64_t>> predict_set_dg(torch::Tensor input, double delta, double gamma, int64_t c);
    std::vector<std::vector<int64_t>> predict_set_size(torch::Tensor input, int64_t size, int64_t c);
    std::vector<std::vector<int64_t>> predict_set_error(torch::Tensor input, double error, int64_t c);
    std::vector<std::vector<int64_t>> predict_set_lac(torch::Tensor input, double error, int64_t c);
    std::vector<std::vector<int64_t>> predict_set_raps(torch::Tensor input, double error, bool rand, double lambda, int64_t k, int64_t c);
    std::vector<std::vector<int64_t>> predict_set_csvphf(torch::Tensor input, double error, bool rand, double lambda, int64_t k, int64_t c);
    std::vector<std::vector<int64_t>> predict_set_crsvphf(torch::Tensor input, double error, bool rand, double lambda, int64_t k); 
    std::vector<double> calibrate_raps(torch::Tensor input, torch::Tensor labels, double error, bool rand, double lambda, int64_t k, int64_t c); 
    std::vector<double> calibrate_csvphf(torch::Tensor input, torch::Tensor labels, double error, bool rand, double lambda, int64_t k, int64_t c);
    std::vector<double> calibrate_crsvphf(torch::Tensor input, torch::Tensor labels, double error, bool rand, double lambda, int64_t k);
    std::vector<std::vector<int64_t>> predict_set(torch::Tensor input, const param& params);
    std::vector<double> calibrate_raps_(torch::Tensor input, torch::Tensor labels, const param& p); 
    std::vector<double> calibrate_raps_hf_(torch::Tensor input, torch::Tensor labels, const param& p);
    std::vector<double> calibrate_csvphf_hf_(torch::Tensor input, torch::Tensor labels, const param& p);
    std::vector<double> calibrate_crsvphf_hf_(torch::Tensor input, torch::Tensor labels, const param& p);
    std::vector<std::vector<int64_t>> lacrsvphf(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> lacusvphf(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> lacsvp(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> rapsrsvphf(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> rapsusvphf(torch::Tensor input, const param& params); 
    std::vector<std::vector<int64_t>> rapssvp(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> csvphfrsvphf(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> crsvphfrsvphf(torch::Tensor input, const param& params); 
    std::vector<std::vector<int64_t>> gsvbop(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> gsvbop_r(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> gsvbop_hf(torch::Tensor input, const param& params);
    std::vector<std::vector<int64_t>> gsvbop_hf_r(torch::Tensor input, const param& params);
    std::tuple<std::vector<int64_t>, double> _gsvbop_hf_r(torch::Tensor input, const param& params, int64_t c, std::vector<int64_t> ystar, double ystar_u, std::vector<int64_t> yhat, double yhat_p, std::priority_queue<QNode> q);
    std::tuple<std::vector<int64_t>, double> min_cov_set_r(torch::Tensor input, std::vector<int64_t> a, const param& params);
    std::tuple<std::vector<int64_t>, double> _min_cov_set_r(torch::Tensor input, std::vector<int64_t> a, int64_t c, std::vector<int64_t> ystar, double ystar_p, std::vector<int64_t> yhat, double yhat_p, std::priority_queue<QNode> q);
    // DP solution
    void init_ca_sr(torch::Tensor input, const std::vector<int64_t>& a, std::vector<HNode*>& q, const param& params);
    void update_ca_sr(torch::Tensor input, QNode& a, std::vector<HNode*>& q, const param& params);
    std::tuple<std::vector<int64_t>, double> min_cov_set_dp(torch::Tensor input, std::vector<int64_t> a, std::vector<HNode*>& q, const param& params);
    // other functions
    void set_hstruct(torch::Tensor hstruct);
};

#endif