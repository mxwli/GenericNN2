#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <fstream>
#include <cassert>
#include <chrono>

#include "NN.h"

using namespace std;

#define MNISTTEST


#ifdef MNISTTEST

vector<NN::vector> X_train, X_test, y_train, y_test;

void read_mnist(string filename, vector<NN::vector>& X, vector<NN::vector>& y, int cutoff) {
	ifstream filereader(filename);
	string dummy; filereader >> dummy; // we dont need the labels
	while(!filereader.eof() && (cutoff == -1 || X.size() < cutoff)) {
		if(X.size()%1000==0) cout << "reading line " << X.size() << "\n";
		string nextline;
		filereader >> nextline; // since there are no spaces in the file
		int val = 0;
		bool first = true;
		vector<double> new_x;
		for(char i: nextline) {
			if(i == ',') {
				if(first) {
					vector<double> tmp(10);
					tmp[val] = 1;
					y.push_back(tmp);
				}
				else new_x.push_back(val/255.0);
				first = false;
				val = 0;
			}
			else {
				val = 10*val+(i-'0');
			}
		}
		new_x.push_back(val);
		vector<double> processed;
		// here we use max pooling
		for(int x = 0; x+1 < 28; x+=2) {
			for(int y = 0; y+1 < 28; y+=2) {
				double mx = 0;
				for(int x1 = x; x1 <= x+1; x1++) for(int y1 = y; y1 <= y+1; y1++)
					mx = max(mx, new_x[x+28*y]);
				processed.push_back(mx);
			}
		}
		X.push_back(NN::vector(processed));
	}
	cout << "closing file" << endl;
	filereader.close();
}

void hitmissratio(const NN::network& net) {
	cout << "\t\tevaluating model" << endl;
	int hit = 0, miss = 0;
	for(int i = 0; i < X_test.size(); i++) {
		vector<double> prediction = net(X_test[i]).back();
		int maxidxpredict = 0, maxidxlabel = 0;
		for(int i2 = 1; i2 < 10; i2++) {
			if(prediction[i2]>prediction[maxidxpredict]) maxidxpredict = i2;
			if(prediction[i2]>y_test[i][maxidxlabel]) maxidxlabel = i2;
		}
		if(maxidxpredict == maxidxlabel) hit++;
		else miss++;
	}
	cout << "\t\thit to miss: " << hit << " : " << miss << endl;
}

void test(string filename) {
	NN::network net({});
	ifstream networkdata(filename);
	cout << "\t\treading network" << endl;
	networkdata >> net;
	networkdata.close();
	cout << "\t\treading test set" << endl;
	read_mnist("data/mnist_test.csv", X_test, y_test, -1);
	assert(X_test.size() == y_test.size());
	hitmissratio(net);
}

int main() {
	int choice = 0;
	cout << "0 for training, 1 for reading from file" << endl;
	cin >> choice;
	if(choice == 1) {
		cout << "enter filename" << endl;
		string name;cin >> name;
		test(name);
		return 0;
	}
	cout << "\t\treading test set\n";
	read_mnist("data/mnist_test.csv", X_test, y_test, -1);
	assert(X_test.size() == y_test.size());
	cout << "\t\treading train set" << endl;
	read_mnist("data/mnist_train.csv", X_train, y_train, -1);
	assert(X_train.size() == y_train.size());
	cout << "\t\tmaking and training network" << endl;
	NN::network net({
		NN::layer("elu", 64, 14*14),
		NN::layer("elu", 32, 64),
		NN::layer("tanh", 10, 32)
	});
	NN::network net2({
		NN::layer("elu", 128, 14*14),
		NN::layer("tanh", 10, 128)
	});
	
	NN::automatic_fit(net, X_train, y_train, "mse", 20, 64, 0.001, hitmissratio, "saves/D1.txt");
}

#endif


#ifdef FOURSINEABTEST

int main() {
	vector<NN::vector> X,y;
	//for(int a = 0; a < 2; a++) for(int b = 0; b < 2; b++) X.push_back(NN::vector({(double)a,(double)b})), y.push_back(NN::vector({(double)(a^b)}));
	for(double a = -1; a < 1; a+=0.1) {
		for(double b = -1; b < 1; b += 0.1) {
			X.push_back(NN::vector({a, b}));
			y.push_back(NN::vector({4*sin(a*b)}));
		}
	}
	NN::network net({
		NN::layer("elu", 5, 2),
		NN::layer("elu", 2, 5),
		NN::layer("linear", 1, 2),
	});
	NN::automatic_fit_old(net, X, y, 400, X.size(), 0.02);
	NN::automatic_fit_old(net, X, y, 400, X.size(), 0.005);
	for(int i = 0; i < X.size(); i++) {
		cout << X[i][0] << " " << X[i][1] << " " << y[i][0] << " : " << net(X[i]).back()[0] << "\n";
	}
	cout << net << "\n";
}

#endif