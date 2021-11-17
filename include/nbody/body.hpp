//
// CUDA implementation.
//
#pragma once

#include <random>
#include <utility>



class BodyPool
{

    // so the movements of bodies are calculated discretely.
    // if after the collision, we do not separate the bodies a little bit, it may
    // results in strange outcomes like infinite acceleration.
    // hence, we will need to set up a ratio for separation.
    

public:
    static constexpr double COLLISION_RATIO = 0.01;
    // provides in this way so that
    // it is easier for you to send a the vector with MPI
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> vx;
    std::vector<double> vy;
    std::vector<double> ax;
    std::vector<double> ay;
    std::vector<double> m;

    // buffer for middle results
    std::vector<double> copy_x;
    std::vector<double> copy_y;
    std::vector<double> copy_vx;
    std::vector<double> copy_vy;

    class Body
    {
        size_t index;
        BodyPool &pool;

        friend class BodyPool;

        Body(size_t index, BodyPool &pool) : index(index), pool(pool) {}

    public:
        double &get_x()
        {
            return pool.x[index];
        }

        double &get_x_copy()
        {
            return pool.copy_x[index];
        }

        double &get_y()
        {
            return pool.y[index];
        }

        double &get_y_copy()
        {
            return pool.copy_y[index];
        }

        double &get_vx()
        {
            return pool.vx[index];
        }

        double &get_vx_copy()
        {
            return pool.copy_vx[index];
        }

        double &get_vy()
        {
            return pool.vy[index];
        }

        double &get_vy_copy()
        {
            return pool.copy_vy[index];
        }

        double &get_ax()
        {
            return pool.ax[index];
        }

        double &get_ay()
        {
            return pool.ay[index];
        }

        double &get_m()
        {
            return pool.m[index];
        }

        double distance_square(Body &that)
        {
            auto delta_x = get_x() - that.get_x();
            auto delta_y = get_y() - that.get_y();
            return delta_x * delta_x + delta_y * delta_y;
        }

        double distance(Body &that)
        {
            return std::sqrt(distance_square(that));
        }

        double delta_x(Body &that)
        {
            return get_x() - that.get_x();
        }

        double delta_y(Body &that)
        {
            return get_y() - that.get_y();
        }

        bool collide(Body &that, double radius)
        {
            return distance_square(that) <= radius * radius;
        }

        // collision with wall
        void handle_wall_collision(double position_range, double radius)
        {
            bool flag = false;
            if (get_x_copy() <= radius)
            {
                flag = true;
                get_x_copy() = radius + radius * COLLISION_RATIO;
                get_vx_copy() = -get_vx_copy();
            }
            else if (get_x_copy() >= position_range - radius)
            {
                flag = true;
                get_x_copy() = position_range - radius - radius * COLLISION_RATIO;
                get_vx_copy() = -get_vx_copy();
            }

            if (get_y_copy() <= radius)
            {
                flag = true;
                get_y_copy() = radius + radius * COLLISION_RATIO;
                get_vy_copy() = -get_vy_copy();
            }
            else if (get_y_copy() >= position_range - radius)
            {
                flag = true;
                get_y_copy() = position_range - radius - radius * COLLISION_RATIO;
                get_vy_copy() = -get_vy_copy();
            }
            if (flag)
            {
                get_ax() = 0;
                get_ay() = 0;
            }
        }

        void update_for_tick(
            double elapse,
            double position_range,
            double radius)
        {
            get_vx_copy() += get_ax() * elapse;
            get_vy_copy() += get_ay() * elapse;

            handle_wall_collision(position_range, radius); // change x & v in buffer "copy"

            get_x_copy() += get_vx_copy() * elapse;
            get_y_copy() += get_vy_copy() * elapse;

            handle_wall_collision(position_range, radius); // change x & v in buffer "copy"

        }
    };

    BodyPool(size_t size, double position_range, double mass_range) : x(size), y(size), vx(size), vy(size), ax(size), ay(size), m(size),
                                                                      copy_x(size), copy_y(size), copy_vx(size), copy_vy(size)
    {
        std::random_device device;
        std::default_random_engine engine{device()};
        std::uniform_real_distribution<double> position_dist{0, position_range};
        std::uniform_real_distribution<double> mass_dist{0, mass_range};

        for (auto &i : x)
        {
            i = position_dist(engine);
        }
        for (auto &i : y)
        {
            i = position_dist(engine);
        }
        for (auto &i : m)
        {
            i = mass_dist(engine);
        }
    }

    Body get_body(size_t index)
    {
        return {index, *this};
    }

    void clear_acceleration()
    {
        ax.assign(m.size(), 0.0);
        ay.assign(m.size(), 0.0);
    }

    size_t size()
    {
        return m.size();
    }

    static void check_and_update(Body i, Body j, double radius, double gravity)
    {
        auto delta_x = i.delta_x(j);
        auto delta_y = i.delta_y(j);
        auto distance_square = i.distance_square(j);
        auto ratio = 1 + COLLISION_RATIO;
        if (distance_square < radius * radius)
        {
            distance_square = radius * radius;
        }
        auto distance = i.distance(j);
        if (distance < radius)
        {
            distance = radius;
        }
        if (i.collide(j, radius))
        {
            auto dot_prod = delta_x * (i.get_vx() - j.get_vx()) + delta_y * (i.get_vy() - j.get_vy());
            auto scalar = 2 / (i.get_m() + j.get_m()) * dot_prod / distance_square;
            i.get_vx_copy() -= scalar * delta_x * j.get_m();
            i.get_vy_copy() -= scalar * delta_y * j.get_m();

            // now relax the distance a bit: after the collision, there must be
            // at least (ratio * radius) between them
            i.get_x_copy() += delta_x / distance * ratio * radius / 2.0;
            i.get_y_copy() += delta_y / distance * ratio * radius / 2.0;
        }
        else
        {
            // update acceleration only when no collision
            auto scalar = gravity / distance_square / distance;
            i.get_ax() -= scalar * delta_x * j.get_m();
            i.get_ay() -= scalar * delta_y * j.get_m();
        }
    }

    void update_for_tick(double elapse,
                         double gravity,
                         double position_range,
                         double radius, size_t localoffset, size_t localsize)
    {
        ax.assign(size(), 0);
        ay.assign(size(), 0);
        // use the status of N-1 bodies to update the v & x in BUFFER "copy"; update the a in vector ax, ay
        for (size_t i = localoffset; i < localoffset + localsize; ++i)
        {
            for (size_t j = 0; j < size(); ++j)
            {
                check_and_update(get_body(i), get_body(j), radius, gravity);
            }
        }
        // update the v & x based on BUFFER to the BUFFER
        for (size_t i = localoffset; i < localoffset + localsize; ++i)
        {
            get_body(i).update_for_tick(elapse, position_range, radius);
        }
    }
};
