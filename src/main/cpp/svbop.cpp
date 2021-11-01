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

struct Softmax : torch::nn::Module {
  // constructor
  Softmax(int64_t D, int64_t K)
      : linear(register_module("linear", torch::nn::Linear(D, K))) {};
  // forward 
  torch::Tensor forward(torch::Tensor input) {
    auto o = linear(input);
    o = torch::nn::functional::softmax(o, torch::nn::functional::SoftmaxFuncOptions(1));
    return o;
  }
  // top-k
  torch::Tensor topk(torch::Tensor input, int64_t k) {
    auto o = linear(input);
    o = torch::nn::functional::softmax(o, torch::nn::functional::SoftmaxFuncOptions(1)); 
    torch::Tensor ind = torch::argsort(o, 1);
    ind = ind.slice(1,0,k,1);
    o = o.gather(1,ind);
    return o;
  }
  // attributes
  torch::nn::Linear linear;
};

/*
Structure which represents a node in some hierarchy
*/
struct HNode : torch::nn::Module {
    torch::nn::Linear estimator {nullptr};
    std::vector<int64_t> y;
    std::vector<HNode*> chn;
    torch::nn::Module *par;
    void addChildNode(int64_t D, std::vector<int64_t> y); 
    torch::Tensor predict(torch::Tensor input, int64_t x_ind, int64_t y_ind);
    torch::Tensor topk(torch::Tensor input, int64_t k);
};

void HNode::addChildNode(int64_t D, std::vector<int64_t> y) {
   // check if leaf or internal node 
    if (this->chn.size() > 0)
    {
        // check if y is a subset of one of the children
        int64_t ind = -1;
        for (int64_t i = 0; i < this->chn.size(); ++i)
        {
            if (std::includes(this->chn[i]->y.begin(), this->chn[i]->y.end(), y.begin(), y.end()) == 1)
            {
                ind = i;
                break;
            }
        }
        if (ind != -1)
            // subset found, hence, recursively pass to child
            this->chn[ind]->addChildNode(D,y);
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
                this->estimator = this->par->register_module(ystr.str(), torch::nn::Linear(D,2));
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

torch::Tensor HNode::predict(torch::Tensor input, int64_t x_ind, int64_t y_ind)
{
    torch::Tensor prob;
    if (this->chn.size() > 1)
    {
        prob = this->estimator->forward(input);
        prob = torch::nn::functional::softmax(prob, torch::nn::functional::SoftmaxFuncOptions(1));
        prob = prob[x_ind][y_ind];
    }
    return prob;
}

std::vector<std::vector<int64_t>> getStruct(int64_t K) {
  std::vector<int64_t> r(K);
  std::iota(r.begin(), r.end(), 0);
  std::vector<std::vector<int64_t>> o;
  std::queue<std::vector<int64_t>> to_process;
  to_process.push(r);
  while (!to_process.empty())
  {
    std::vector<int64_t> el {to_process.front()};
    to_process.pop();
    if (el.size() > 1)
    {
      // split and add to process Q
      std::vector<int64_t> left(el.begin(),el.begin()+el.size()/2);
      std::vector<int64_t> right(el.begin()+ el.size()/2,el.end());
      to_process.push(left);
      to_process.push(right);
    }
    // add processed element to output vector
    o.push_back(el);
  }
  return o;
}

struct HSoftmax : torch::nn::Module {
  // constructor
  HSoftmax(int64_t D, int64_t K) {
    // first create root node 
    this->root = new HNode();
    // also create struct
    std::vector<std::vector<int64_t>> hstruct {getStruct(K)};
    // now construct tree
    this->root->y = hstruct[0];
    this->root->chn = {};
    this->root->par = this;
    for (int64_t i = 1; i < hstruct.size(); ++i)
        this->root->addChildNode(D, hstruct[i]);    
  }

  // forward 
  torch::Tensor forward(torch::Tensor input, std::vector<int64_t> ind) {
    int64_t batch_size {input.size(0)};
    std::vector<torch::Tensor> probs;
    // run over each sample in batch
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    for (int64_t bi = start; bi < end; bi++)
    //for (int64_t bi=0;bi<batch_size;++bi)
    {
      // begin at root
      HNode* visit_node = this->root;
      torch::Tensor prob = torch::ones(1);
      while (!visit_node->chn.empty())
      {
          int64_t found_ind {-1};
          for (int64_t i = 0; i<visit_node->chn.size(); ++i)
          { 
              if (std::count(visit_node->chn[i]->y.begin(), visit_node->chn[i]->y.end(), ind[bi]))
              {
                  found_ind = i;
                  break;
              }  
          }
          if (found_ind != -1)
          {
              prob = prob*visit_node->predict(input, bi, found_ind);
              visit_node = visit_node->chn[found_ind];
          }
      }
      probs.push_back(prob);
    }
    });
    return torch::stack(probs);
  }

  // top-k
  torch::Tensor topk(torch::Tensor input, int64_t k) {
    std::vector<int64_t> t {1,1};
    return torch::tensor(t);
  }

  // attributes
  HNode* root;
};

/*
int main_old() {
  // some constants
  int64_t K {100};
  int64_t D {100};
  int64_t BS {2};
  // start analysis
  std::cout << "ANALYSIS SOFTMAX" << std::endl;
  torch::Tensor tensor = torch::randn({BS, D});;
  Softmax n1 = Softmax(D, K);
  auto t1 = std::chrono::high_resolution_clock::now();
  torch::Tensor o = n1.forward(tensor);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "Calculated o in " << time << " ms" << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "ANALYSIS H-SOFTMAX" << std::endl;
  tensor = torch::randn({BS, D});
  HSoftmax n2 = HSoftmax(D, K);
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_int_distribution<int64_t> dist {0, K-1};
  auto gen = [&dist, &mersenne_engine](){return dist(mersenne_engine);};
  std::vector<int64_t> ind(BS);
  generate(begin(ind), end(ind), gen);
  t1 = std::chrono::high_resolution_clock::now();
  float_t learning_rate = 1e-4;
  torch::optim::SGD optimizer(n2.parameters(),torch::optim::SGDOptions(learning_rate).momentum(0.5));
  n2.zero_grad();
  o = n2.forward(tensor, ind);
  torch::Tensor loss = torch::binary_cross_entropy(o, torch::ones({BS,1}));
  std::cout << loss << std::endl;
  loss.backward();
  optimizer.step();
  t2 = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "Calculated o in " << time << " ms" << std::endl;
  std::cout << "NUMBER OF PARAMS: " << n2.parameters().size() << std::endl;
}
*/

void test_forward(int64_t K, int64_t D, int64_t BS) {
  std::cout << "ANALYSIS BACKWARD SOFTMAX" << std::endl;
  torch::Tensor tensor = torch::randn({BS, D});
  Softmax n1 = Softmax(D, K);
  auto t1 = std::chrono::high_resolution_clock::now();
  torch::Tensor o = n1.forward(tensor);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "Calculated o in " << time << " ms" << std::endl;
  std::cout << std::endl;
  std::cout << std::endl; 

  std::cout << "ANALYSIS BACKWARD H-SOFTMAX" << std::endl;
  tensor = torch::randn({BS, D});
  HSoftmax n2 = HSoftmax(D, K);
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_int_distribution<int64_t> dist {0, K-1};
  auto gen = [&dist, &mersenne_engine](){return dist(mersenne_engine);};
  std::vector<int64_t> ind(BS);
  generate(begin(ind), end(ind), gen);
  t1 = std::chrono::high_resolution_clock::now();
  o = n2.forward(tensor, ind);
  t2 = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "Calculated o in " << time << " ms" << std::endl;
  std::cout << "NUMBER OF PARAMS: " << n2.parameters().size() << std::endl;
  std::cout << std::endl;
  std::cout << std::endl; 
}

void test_backward(int64_t K, int64_t D, int64_t BS) {
  std::cout << "ANALYSIS BACKWARD SOFTMAX" << std::endl;
  torch::Tensor tensor = torch::randn({BS, D});
  Softmax n1 = Softmax(D, K);
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_int_distribution<int64_t> dist {0, K-1};
  auto gen = [&dist, &mersenne_engine](){return dist(mersenne_engine);};
  std::vector<int64_t> ind(BS);
  generate(begin(ind), end(ind), gen);
  torch::Tensor ind_t = torch::tensor(ind);
  float_t learning_rate = 1e-4;
  torch::optim::SGD optimizer(n1.parameters(),torch::optim::SGDOptions(learning_rate).momentum(0.5));
  n1.zero_grad();
  torch::Tensor o = n1.forward(tensor);
  auto t1 = std::chrono::high_resolution_clock::now();
  torch::Tensor loss = torch::nll_loss(o, ind_t);
  loss.backward();
  optimizer.step();
  auto t2 = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "Calculated loss in " << time << " ms" << std::endl;
  std::cout << std::endl;
  std::cout << std::endl; 

  std::cout << "ANALYSIS BACKWARD H-SOFTMAX" << std::endl;
  tensor = torch::randn({BS, D});
  HSoftmax n2 = HSoftmax(D, K);
  learning_rate = 1e-4;
  n2.zero_grad();
  o = n2.forward(tensor, ind);
  t1 = std::chrono::high_resolution_clock::now();
  loss = torch::binary_cross_entropy(o, torch::ones({BS,1}));
  loss.backward();
  optimizer.step();
  t2 = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "Calculated loss in " << time << " ms" << std::endl;
  std::cout << "NUMBER OF PARAMS: " << n2.parameters().size() << std::endl;
  std::cout << std::endl;
  std::cout << std::endl; 
}

void test_topk(int64_t K, int64_t D, int64_t BS, int64_t k)
{
  std::cout << "ANALYSIS TOPK SOFTMAX" << std::endl;
  torch::Tensor tensor = torch::randn({BS, D});
  Softmax n1 = Softmax(D, K);
  auto t1 = std::chrono::high_resolution_clock::now();
  torch::Tensor o = n1.topk(tensor, k);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "Calculated o in " << time << " ms" << std::endl;
  std::cout << std::endl;
  std::cout << std::endl; 

  std::cout << "ANALYSIS TOPK H-SOFTMAX" << std::endl;
  tensor = torch::randn({BS, D});
  HSoftmax n2 = HSoftmax(D, K);
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_int_distribution<int64_t> dist {0, K-1};
  auto gen = [&dist, &mersenne_engine](){return dist(mersenne_engine);};
  std::vector<int64_t> ind(BS);
  generate(begin(ind), end(ind), gen);
  t1 = std::chrono::high_resolution_clock::now();
  o = n2.topk(tensor, k);
  t2 = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast <std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "Calculated o in " << time << " ms" << std::endl;
  std::cout << "NUMBER OF PARAMS: " << n2.parameters().size() << std::endl;
  std::cout << std::endl;
  std::cout << std::endl; 
}

int main() {
  test_forward(10000, 100, 8);
  test_backward(10000, 100, 8);
  test_topk(10000, 100, 8, 5000);
}

PYBIND11_MODULE(hsoftmax_cpp, m) {
   torch::python::bind_module<HSoftmax>(m, "HSoftmax")
     .def(py::init<int64_t, int64_t>())
     .def("forward", &HSoftmax::forward)
     .def("topk", &HSoftmax::topk);
}