/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <iomanip>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 500;

	default_random_engine gen;

	normal_distribution<double> distribution_x(x, std[0]);
	normal_distribution<double> distribution_y(y, std[1]);
	normal_distribution<double> distribution_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		Particle particle;
		particle.id = i;
		particle.x = distribution_x(gen);
		particle.y = distribution_y(gen);
		particle.theta = distribution_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double x_f, y_f, theta_f;
	default_random_engine gen;
	for (int i = 0; i < num_particles; ++i) {
		if (fabs(yaw_rate) > 1e-4) {
			theta_f = particles[i].theta + yaw_rate*delta_t;
			x_f = particles[i].x + velocity/yaw_rate*(sin(theta_f)-sin(particles[i].theta));
			y_f = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta)-cos(theta_f));
		}
		else {
			theta_f = particles[i].theta;
			x_f = particles[i].x + velocity*delta_t*cos(particles[i].theta);
			y_f = particles[i].y + velocity*delta_t*sin(particles[i].theta);
		}

		normal_distribution<double> distribution_x_f(x_f, std_pos[0]);
		normal_distribution<double> distribution_y_f(y_f, std_pos[1]);
		normal_distribution<double> distribution_theta_f(theta_f, std_pos[2]);

		particles[i].x = distribution_x_f(gen);
		particles[i].y = distribution_y_f(gen);
		particles[i].theta = distribution_theta_f(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	//pre-comupted coefficients
	double coeff1 = 0.5/M_PI/std_landmark[0]/std_landmark[1];
	double coeff2 = 0.5/std_landmark[0]/std_landmark[0];
	double coeff3 = 0.5/std_landmark[1]/std_landmark[1];

	for (int i = 0; i < num_particles; i++) {
		particles[i].weight = 1.0; //weight reset
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;
		for (int j = 0; j < observations.size(); j++) {
			// Coordinate transform
			double cos_theta = cos(particles[i].theta);
			double sin_theta = sin(particles[i].theta);
			double x_new, y_new;
			x_new = particles[i].x + observations[j].x*cos_theta - observations[j].y*sin_theta;
			y_new = particles[i].y + observations[j].x*sin_theta + observations[j].y*cos_theta;
			// DataAssociation Step
			int id_min = 0;
			double dist_min = 10000.0; //some large enough number
			for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
				//use infinite-norm instead of two-norm beacuse it is cheaper (from Udacity forum)!
				double dist_pt2landmark_x = fabs(particles[i].x-map_landmarks.landmark_list[k].x_f);
				double dist_pt2landmark_y = fabs(particles[i].y-map_landmarks.landmark_list[k].y_f);
				// double dist_pt2landmark = dist(particles[i].x,particles[i].y,map_landmarks.landmark_list[k].x_f,map_landmarks.landmark_list[k].y_f);
				if (fmax(dist_pt2landmark_x,dist_pt2landmark_y) < sensor_range){
					double dist_obs2landmark = dist(x_new,y_new,map_landmarks.landmark_list[k].x_f,map_landmarks.landmark_list[k].y_f);
					if (dist_obs2landmark < dist_min){
						id_min = map_landmarks.landmark_list[k].id_i;
						dist_min = dist_obs2landmark;
					}
				}
			}
			// Update weights with likelihood functions Pi_j p(y_j = y_j|x_i)
			if (id_min > 0){
				double mu_x = map_landmarks.landmark_list[id_min-1].x_f;
				double mu_y = map_landmarks.landmark_list[id_min-1].y_f;
				double liklihood = coeff1 * exp(-pow(x_new-mu_x,2.0)*coeff2-pow(y_new-mu_y,2.0)*coeff3);
				particles[i].weight *= liklihood;
				associations.push_back(id_min);
				sense_x.push_back(x_new);
				sense_y.push_back(y_new);
			}
		}
		//if no valid observations then the particle is discarded
		if (associations.size() == 0) particles[i].weight = 0;

		particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
		weights[i] = particles[i].weight;
	}


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> distribution_posterior(weights.begin(),weights.end());
	vector<Particle> new_particles;
	for (int i = 0; i < num_particles; ++i) {
		new_particles.push_back(particles[distribution_posterior(gen)]);
	}
	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
		particle.associations.clear();
		particle.sense_x.clear();
		particle.sense_y.clear();

		particle.associations= associations;
	 	particle.sense_x = sense_x;
	 	particle.sense_y = sense_y;

	 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
