#include <vector>

#include <Maze/Mazes/MazeMA.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"

namespace Environments {
    using enum EFieldState;

    MazeMA::MazeMA() : IMazeEnvironment(
    11 * 4,
    {
        .mazeWidth = 11,
        .mazeHeight = 7,
        .goalState = {1, 7},
        .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, OBSTACLE, CORRIDOR, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, PRIZE,    OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, CORRIDOR, OBSTACLE, OBSTACLE, CORRIDOR, CORRIDOR, OBSTACLE, OBSTACLE },
                    { OBSTACLE, OBSTACLE, CORRIDOR, OBSTACLE, OBSTACLE, OBSTACLE, CORRIDOR, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, CORRIDOR, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                    //            {1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1},
                    //            {1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1},
                    //            {1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1},
                    //            {1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1},
                    //            {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1},
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            }
    }) {}
}