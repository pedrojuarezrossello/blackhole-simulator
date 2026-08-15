#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>

#include "integrator.h"
#include "message.h"
#include "message_queue.h"
#include "particle.h"
#include "renderer.h"
#include "utils.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>

fixed_sized_message_queue<message, true> data_queue;

namespace {
std::filesystem::path default_data_path(const char* argv0) {
    const auto exe_dir = std::filesystem::absolute(argv0).parent_path();
    const auto candidates = {
        exe_dir / "data.txt",
        exe_dir / "../src/data.txt",
        std::filesystem::current_path() / "data.txt",
        std::filesystem::current_path() / "src/data.txt"
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate))
            return std::filesystem::weakly_canonical(candidate);
    }
    throw std::runtime_error("Could not find data.txt. Pass its path as the second command-line argument.");
}
}

int main(int argc, char* argv[]) {
    try {
        const float param = argc > 1 ? std::strtof(argv[1], nullptr) : 0.3f;
        const auto data_path = argc > 2 ? std::filesystem::path(argv[2]) : default_data_path(argv[0]);

        // The uploaded project currently contains Kerr initial data. Keep the
        // simulation selection here explicit so the rendering layer remains metric-agnostic.
        initial_particle_data<kerr> particle_data(data_path.string());
        kerr_integrator solver(param, particle_data);

        constexpr unsigned width = 2400;
        constexpr unsigned height = 1200;

        // Request a compatibility OpenGL context because the renderer deliberately
        // uses SFML only for window/context management and OpenGL's simple fixed
        // pipeline for the visual port. SFML will negotiate the closest supported context.
        sf::ContextSettings context_settings;
        context_settings.depthBits = 24;
        context_settings.stencilBits = 8;
        context_settings.antialiasingLevel = 4;
        context_settings.majorVersion = 2;
        context_settings.minorVersion = 1;

        sf::RenderWindow window(
            sf::VideoMode({width, height}),
            "Kerr black hole simulation",
            sf::Style::Default,
            sf::State::Windowed,
            context_settings);

        window.setVerticalSyncEnabled(true);
        window.setActive(true);

        const size_t N = particle_data.initial_radii.size();
        particle_set particles(N, param, kerr{});
        Renderer renderer(width, height, particles.ev_hor);

        std::jthread integrator_thread([&solver]() {
            solver.rock_n_roll();
        });

        while (window.isOpen()) {
            while (const std::optional event = window.pollEvent()) {
                if (event->is<sf::Event::Closed>())
                    window.close();
                else if (const auto* resized = event->getIf<sf::Event::Resized>())
                    renderer.resize(resized->size.x, resized->size.y);
            }

            // The integrator produces snapshots on its own thread. pop() blocks
            // until the next complete snapshot is available, just as the old
            // OpenFrameworks update() did.
            const auto message = data_queue.pop();
            particles.update(std::move(message.xs), std::move(message.ys),
                             std::move(message.zs), std::move(message.radii),
                             std::move(message.states));

            renderer.draw(particles);
            window.display();
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}
