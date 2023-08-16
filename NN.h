#ifndef NN_H
#define NN_H

#include <vector>
#include <functional>
#include <numeric>
#include <string>
#include <cmath>
#include <random>
#include <istream>
#include <ostream>
#include <iostream>
#include <string>
#include <chrono>

namespace NN {
	inline std::mt19937_64 rng;

	typedef std::vector<double> stdvec;
	typedef std::vector<std::vector<double>> stdmat;

	class vector {
		std::vector<double> vec;
	public:
		inline vector(): vec() {}
		inline vector(stdvec v): vec(v) {}
		inline operator stdvec () const {
			return vec;
		}
		inline double operator[](int i) const {
			return vec[i];
		}
		inline friend std::ostream& operator<<(std::ostream& os, const vector& v) {
			os << v.vec.size() << "\n";
			for(const auto& i: v.vec) os << i << " ";
			os << "\n";
			return os;
		}
		inline friend std::istream& operator>>(std::istream& is, vector& v) {
			int N; is >> N;
			for(int i = 0; i < N; i++) {
				double a; is >> a;
				v.vec.push_back(a);
			}
			return is;
		}
		inline vector(std::size_t size, std::function<double(int)> constructor): vec(size) {
			for(std::size_t i = 0; i < size; i++) vec[i] = constructor(i);
		}
		inline vector operator+(const vector& v) const {
			return vector(vec.size(), [this, v](int i) -> double {return vec[i]+v.vec[i];});
		}
		inline vector operator-(const vector& v) const {
			return vector(vec.size(), [this, v](int i) -> double {return vec[i]-v.vec[i];});
		}
		inline vector operator*(const vector& v) const {
			return vector(vec.size(), [this, v](int i) -> double {return vec[i]*v.vec[i];});
		}
		inline vector operator/(const vector& v) const {
			return vector(vec.size(), [this, v](int i) -> double {return vec[i]/v.vec[i];});
		}
		inline vector map(std::function<double(double)> f) const {
			return vector(vec.size(), [this, f](int i) -> double {return f(vec[i]);});
		}
		inline vector operator*(double scale) const {
			return map([scale](double v) -> double {return v*scale;});
		}
	};

	class matrix {
		stdmat mat;
	public:
		std::size_t N, M;
		inline matrix(): mat() {}
		inline matrix(stdmat m): mat(m), N(m.size()), M(m[0].size()) {}
		inline matrix(stdvec v, bool vert=true): mat(v.size()), N(v.size()), M(1) {
			for(std::size_t i = 0; i < v.size(); i++) mat[i] = {v[i]};
		}
		inline operator stdmat () const {
			return mat;
		}
		inline operator stdvec () const { //flatten matrix into a vector
			return vector(N*M, [this](int i) -> double {return mat[i/M][i%M];});
		}
		inline stdvec operator[](int idx) const {
			return mat[idx];
		}
		inline friend std::ostream& operator<<(std::ostream& os, const matrix& m) {
			os << m.N << " " << m.M << "\n";
			for(const auto& i: m.mat) {
				for(const auto& i2: i) 
					os << i2 << " ";
				os << "\n";
			}
			return os;
		}
		inline friend std::istream& operator>>(std::istream& is, matrix& m) {
			is >> m.N >> m.M;
			for(std::size_t i = 0; i < m.N; i++) {
				m.mat.push_back(stdvec(m.M));
				for(auto& i2: m.mat[i]) is >> i2;
			}
			return is;
		}
		inline matrix(std::size_t N, std::size_t M, std::function<double(int, int)> constructor): mat(N, stdvec(M)), N(N), M(M) {
			for(std::size_t i = 0; i < N; i++)
				for(std::size_t i2 = 0; i2 < M; i2++)
					mat[i][i2] = constructor(i, i2);
		}
		inline matrix(std::size_t N, std::size_t M): mat(N, stdvec(M)), N(N), M(M) {}
		inline matrix operator+(const matrix& m) const {
			return matrix(N, M, [this, m](int x, int y) -> double {return mat[x][y]+m.mat[x][y];});
		}
		inline matrix operator-(const matrix& m) const {
			return matrix(N, M, [this, m](int x, int y) -> double {return mat[x][y]-m.mat[x][y];});
		}
		inline matrix operator*(const matrix& m) const {
			return matrix(N, M, [this, m](int x, int y) -> double {return mat[x][y]*m.mat[x][y];});
		}
		inline matrix operator/(const matrix& m) const{
			return matrix(N, M, [this, m](int x, int y) -> double {return mat[x][y]/m.mat[x][y];});
		}
		inline matrix operator&(const matrix& m) const { // matrix multiplication is reserved with the & operator
			stdmat ret(N, stdvec(m.M, 0));
			for(std::size_t x = 0; x < N; x++)
				for(std::size_t i = 0; i < M; i++)
					for(std::size_t y = 0; y < m.M; y++)
						ret[x][y] += mat[x][i]*m.mat[i][y];
			return matrix(ret);
			/*
			return matrix(N, m.M, [this, m](int x, int y) -> double {
				double sum = 0;
				for(size_t i = 0; i < M; i++) sum += mat[x][i]*m.mat[i][y];
				return sum;
			});
			*/
		}
		inline matrix map(std::function<double(double)> f) const {
			return matrix(N, M, [this, f](int x, int y) -> double {return f(mat[x][y]);});
		}
		inline matrix operator*(double scale) const {
			return map([scale](double v) -> double {return v*scale;});
		}
	};
	
	class act_function {
	public:
		std::string name;
		std::function<double(double)> f, df;
		inline act_function(): name("linear") {
			f = [](double x) -> double {return x;};
			df = [](double x) -> double {return 1;};
		}
		inline act_function(std::string name): name(name) {
			if(name == "linear") {
				f = [](double x) -> double {return x;};
				df = [](double x) -> double {return 1;};
			}
			else if (name == "tanh") {
				f = [](double x) -> double {return 0.5*std::tanh(x)+0.5;};
				df = [](double x) -> double {return std::pow(1.0/std::cosh(x),2)/2;};
			}
			else if (name == "relu") {
				f = [](double x) -> double {return std::max(x, 0.2*x);};
				df = [](double x) -> double {return x<0?0.2:1;};
			}
			if(name == "elu") {
				f = [](double x) -> double {return x>0?x:exp(x)-1;};
				df = [](double x) -> double {return x>0?1:exp(x);};
			}
		}
		inline double rand(int size, int prev_size) const {
			if(name == "linear") return 0;
			if(name == "tanh") {
				return std::normal_distribution<double>(0, std::sqrt(2.0/(size+prev_size)))(rng);
			}
			if(name == "relu") return std::normal_distribution<double>(0, std::sqrt(2.0/prev_size))(rng);
			if(name == "elu") return std::normal_distribution<double>(0, std::sqrt(2.0/prev_size))(rng);
			return 0;
		}
		inline friend std::ostream& operator<<(std::ostream& os, const act_function& act) {
			os << act.name << "\n";
			return os;
		}
		inline friend std::istream& operator>>(std::istream& is, act_function& act) {
			is >> act.name;
			act = act_function(act.name);
			return is;
		}
	};
	
	class layer {
	public:
		act_function activation;
		std::size_t size, prev_size;
		vector biases;
		matrix weight;
		inline layer(): activation(), biases(), weight() {}
		inline layer(std::string activation_name, std::size_t size, std::size_t prev_size):
			activation(activation_name), size(size), prev_size(prev_size) {
			biases = vector(size, [this, size, prev_size](int i) -> double { return activation.rand(size, prev_size);});
			weight = matrix(size, prev_size, [this, size, prev_size](int x, int y) -> double {return activation.rand(size, prev_size);});
		}
		inline layer(act_function activation, vector b, matrix m): activation(activation), biases(b), weight(m) {
			size = ((stdvec)b).size();
			prev_size = ((stdmat)m)[0].size();
		}
		inline friend std::ostream& operator<<(std::ostream& os, const layer& l) {
			os << l.size << " " << l.prev_size << "\n";
			os << l.weight << l.biases << l.activation;
			return os;
		}
		inline friend std::istream& operator>>(std::istream& is, layer& l) {
			is >> l.size >> l.prev_size;
			is >> l.weight >> l.biases >> l.activation;
			return is;
		}
		inline vector operator()(vector v) const {
			return (vector(weight&matrix(v))+biases).map(activation.f);
		}
		inline layer operator+(const layer& l) const {
			return layer(activation, biases+l.biases, weight+l.weight);
		}
		inline layer operator-(const layer& l) const {
			return layer(activation, biases-l.biases, weight-l.weight);
		}
		inline layer operator*(const layer& l) const {
			return layer(activation, biases*l.biases, weight*l.weight);
		}
		inline layer operator/(const layer& l) const {
			return layer(activation, biases/l.biases, weight/l.weight);
		}
		inline layer map(std::function<double(double)> f) {
			return layer(activation, biases.map(f), weight.map(f));
		}
		inline layer operator*(double scale) const {
			return layer(activation, biases*scale, weight*scale);
		}
	};

	typedef std::vector<vector> network_output;

	class network {
	public:
		std::vector<layer> layers;
		inline network(std::vector<layer> layers): layers(layers) {}
		inline friend std::ostream& operator<<(std::ostream& os, const network& n) {
			os << n.layers.size() << "\n";
			for(const auto& i: n.layers) os << i;
			return os;
		}
		inline friend std::istream& operator>>(std::istream& is, network& n) {
			int N; is >> N;
			for(int i = 0; i < N; i++) {
				layer newlayer;
				is >> newlayer;
				n.layers.push_back(newlayer);
			}
			return is;
		}
		inline network_output operator()(vector v) const {
			std::vector<vector> ret;
			for(const layer& l: layers) {
				v = l(v);
				ret.push_back(v);
			}
			return ret;
		}
		inline network operator+(const network& n) const {
			std::vector<layer> ret;
			for(std::size_t i = 0; i < layers.size(); i++)
				ret.push_back(layers[i]+n.layers[i]);
			return ret;
		}
		inline network operator-(const network& n) const {
			std::vector<layer> ret;
			for(std::size_t i = 0; i < layers.size(); i++)
				ret.push_back(layers[i]-n.layers[i]);
			return ret;
		}
		inline network operator*(const network& n) const {
			std::vector<layer> ret;
			for(std::size_t i = 0; i < layers.size(); i++)
				ret.push_back(layers[i]*n.layers[i]);
			return ret;
		}
		inline network operator/(const network& n) const {
			std::vector<layer> ret;
			for(std::size_t i = 0; i < layers.size(); i++)
				ret.push_back(layers[i]/n.layers[i]);
			return ret;
		}
		inline network map(std::function<double(double)> f) {
			std::vector<layer> ret;
			for(std::size_t i = 0; i < layers.size(); i++)
				ret.push_back(layers[i].map(f));
			return ret;
		}
		inline network operator*(double scale) const {
			std::vector<layer> ret;
			for(std::size_t i = 0; i < layers.size(); i++)
				ret.push_back(layers[i]*scale);
			return ret;
		}
		inline network gradient(vector input, network_output activations, vector target, std::function<double(int)> dloss) const {
			network ret({});
			vector diff = vector((stdvec(target)).size(), dloss);
			for(int i = ((int)layers.size())-1; i >= 0; i--) {
				diff = diff * (activations[i].map(layers[i].activation.df));
				ret.layers.push_back(layer(
					layers[i].activation,
					diff,
					matrix(layers[i].size, layers[i].prev_size,
						[i, diff, activations, input](int x, int y) -> double {
							return i==0?(diff[x]*input[y]):(diff[x]*activations[i-1][y]);
						})
				));
					//auto start = std::chrono::high_resolution_clock::now();
					//std::cout << "\t" << ((stdvec)diff).size() << " " << layers[i].weight.N << " " << layers[i].weight.M << "\n";
				diff = vector(matrix(stdmat({stdvec(diff)}))&layers[i].weight); // here, & denotes the matrix multiplication operation
					//auto end = std::chrono::high_resolution_clock::now();
					//std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "\n";
			}
			std::reverse(ret.layers.begin(), ret.layers.end());

			return ret;
		}
		inline network get_zero_gradient() const {
			return operator*(0);
		}
	};
	typedef network gradient; // networks and gradients are internally the same thing.

	inline double MSE(const vector& A, const vector& B) {
		double sum = 0;
		for(std::size_t i = 0; i < ((stdvec)A).size(); i++) sum += (A[i]-B[i])*(A[i]-B[i]);
		return sum/((stdvec)A).size();
	}

	inline void automatic_fit_old(network& net, std::vector<vector> X, std::vector<vector> y, int iterations, int batch_size, double learning_rate) {
		// batch size -1 means to train with *all* of the data each iteration
		gradient var_m = net.get_zero_gradient();
		gradient var_s = net.get_zero_gradient();
		double var_beta_1 = 0.9, var_beta_2 = 0.999;
		for(int _A = 1; _A <= iterations; _A++) {
			gradient total_gradient = net.get_zero_gradient();
			double avg_mse = 0;
			std::vector<int> indecies(X.size());
			iota(indecies.begin(), indecies.end(), 0);
			shuffle(indecies.begin(), indecies.end(), rng);
			for(int i = 0; i < batch_size && i < X.size(); i++) {
				network_output cur_output = net(X[indecies[i]]);
				vector target = y[indecies[i]];
				total_gradient = total_gradient + net.gradient(X[indecies[i]], cur_output, y[indecies[i]], [cur_output, target](int idx) -> double {
					return (cur_output.back()[idx]-target[idx])*2;
				});
				avg_mse += MSE(cur_output.back(), y[indecies[i]])/batch_size;
			}
			total_gradient = total_gradient*(1.0/X.size());
			var_m = (var_m * var_beta_1) - (total_gradient * (1-var_beta_1));
			var_s = (var_s * var_beta_2) + ((total_gradient*total_gradient)*(1-var_beta_2));
			gradient var_m_cap = var_m*(1.0/(1-std::pow(var_beta_1, 4*_A)));
			gradient var_s_cap = var_s*(1.0/(1-std::pow(var_beta_2, 4*_A)));
			net = net + var_m_cap / var_s_cap.map([](double x) -> double {return std::sqrt(x+1e-7);}) * learning_rate; //ADAM optimizer
			std::cout << "average MSE for iteration " << _A << " \t:\t " << avg_mse << "\n";
		}
	}

	typedef std::function<double(vector output, vector target)> lossfunctype;
	typedef std::function<std::function<double(int)>(const vector& cur_output, const vector& target)> dlossfunctype;

	inline void get_loss_functions(std::string loss, lossfunctype& loss_function, dlossfunctype& dloss_function) {
		if(loss == "mse") {
			loss_function = [](const vector& A, const vector& B) -> double {
				double sum = 0;
				for(std::size_t i = 0; i < ((stdvec)A).size(); i++) sum += (A[i]-B[i])*(A[i]-B[i]);
				return sum/((stdvec)A).size();
			};
			dloss_function = [](const vector& cur_output, const vector& target) -> std::function<double(int)> {
				return [cur_output, target](int idx) -> double {
			    	return (cur_output[idx] - target[idx]) * 2 / ((stdvec)target).size();
				};
			};
		}
		if (loss == "mae") {
			loss_function = [](const vector& A, const vector& B) -> double {
			    double sum = 0;
			    for (std::size_t i = 0; i < ((stdvec)A).size(); i++) {
					sum += std::abs(A[i] - B[i]);
				}
			    return sum / ((stdvec)A).size();
			};
			dloss_function = [](const vector& cur_output, const vector& target) -> std::function<double(int)> {
			    return [cur_output, target](int idx) -> double {
					return ((cur_output[idx] - target[idx]) > 0 ? 1 : -1) / ((stdvec)target).size();
			    };
			};
		}
		if (loss == "cross entropy") {
			loss_function = [](const vector& A, const vector& B) -> double {
			    double sum = 0;
			    for (std::size_t i = 0; i < ((stdvec)A).size(); i++) {
					if(B[i] != 0)
						sum -= B[i] * std::max(-100.0, std::log(std::clamp(A[i], 1e-3, 1-1e-3)));
				}
			    return sum / ((stdvec)A).size();
			};
			dloss_function = [](const vector& cur_output, const vector& target) -> std::function<double(int)> {
				return [cur_output, target](int idx) -> double {
			    	return -target[idx] / std::clamp(cur_output[idx], 1e-3, 1-1e-3)/((stdvec)target).size();
				};
			};
		}
	}
	/*
	net: network to be trained
	X_train: training data
	y_train: training labels
	loss: string representing loss choice: ["mse", "mae", "cross entropy"]
	metric: called using network once per epoch
	savefile: if not empty, network content is saved in filename
	*/
	inline void automatic_fit(network& net, std::vector<vector> X_train, std::vector<vector> y_train, 
		std::string loss, int epochs, int batch_size, double learning_rate, std::function<void(const network&)> metric, std::string savefile) {
		gradient var_m = net.get_zero_gradient();
		gradient var_s = net.get_zero_gradient();
		double var_beta_1 = 0.9, var_beta_2 = 0.999;
		lossfunctype loss_function;
		dlossfunctype dloss_function;
		get_loss_functions(loss, loss_function, dloss_function);
		for(int epoch_number = 1; epoch_number <= epochs; epoch_number++) {
			std::cout << "-----------------------------" << std::endl;
			gradient total_gradient = net.get_zero_gradient();
			double average_loss = 0;
			std::vector<int> indecies(X_train.size());
			iota(indecies.begin(), indecies.end(), 0);
			shuffle(indecies.begin(), indecies.end(), rng);
			for(int batch_number = 0; batch_size*(batch_number+1) <= X_train.size(); batch_number++) {
				double newal = 0;
				for(int i = 0; i < batch_size; i++) {
					int cidx = indecies[batch_number*batch_size+i];
					vector input = X_train[cidx];
					network_output cur_output = net(X_train[cidx]);
					vector target = y_train[cidx];
					newal += loss_function(cur_output.back(), target);
					total_gradient = total_gradient + net.gradient(input, cur_output, target, dloss_function(cur_output.back(), target));
				}
				newal /= batch_size;
				average_loss += newal;
				total_gradient = total_gradient * (1.0/batch_size);
				var_m = (var_m * var_beta_1) - (total_gradient * (1-var_beta_1));
				var_s = (var_s * var_beta_2) + ((total_gradient*total_gradient)*(1-var_beta_2));
				gradient var_m_cap = var_m*(1.0/(1-std::pow(var_beta_1, 4*epoch_number)));
				gradient var_s_cap = var_s*(1.0/(1-std::pow(var_beta_2, 4*epoch_number)));
				net = net + var_m_cap / var_s_cap.map([](double x) -> double {return std::sqrt(x+1e-7);}) * learning_rate; //ADAM optimizer
				std::cout << "Epoch " << epoch_number << "\t Batch " << batch_number << "\t avg loss " << newal << std::endl;
				if(batch_number%10==0) {
					if(savefile.size() > 0) {
						std::ofstream save(savefile);
						save << net;
						save.close();
					}
				}
			}
			average_loss /= X_train.size()/batch_size;
			std::cout << "\tEpoch " << epoch_number << " " << loss << ": " << average_loss << std::endl;
			metric(net);
		}
	}
}

#endif