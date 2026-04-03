#include <vector>

#include <Maze/Mazes/MazeB.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"

namespace Environments {
    using enum EFieldState;

    MazeB::MazeB() : IMazeEnvironment(
    8 * 4,
    {
        .mazeWidth = 8,
        .mazeHeight = 8,
        .goalState = {2, 6},
        .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, PRIZE,    OBSTACLE },
                    { OBSTACLE, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, CORRIDOR, CORRIDOR, OBSTACLE, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    //            {1, 1, 1, 1, 1, 1, 1, 1},
                    //            {1, 0, 0, 0, 0, 1, 1, 1},
                    //            {1, 0, 0, 1, 0, 1, 2, 1},
                    //            {1, 1, 0, 1, 0, 0, 0, 1},
                    //            {1, 1, 0, 0, 1, 0, 0, 1},
                    //            {1, 0, 1, 0, 0, 1, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 1, 1, 1, 1, 1, 1, 1},
            },
    }) {}
}