#include <vector>

#include <Maze/Mazes/Maze7.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"

namespace Environments {
    using enum EFieldState;

    Maze7::Maze7() : IMazeEnvironment(
    7 * 4,
    {
        .mazeWidth = 5,
        .mazeHeight = 7,
        .goalState = {5, 1},
        .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE },
                    { OBSTACLE, PRIZE,    OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE }
                    //            {1, 1, 1, 1, 1 },
                    //            {1, 0, 0, 0, 1 },
                    //            {1, 0, 1, 0, 1 },
                    //            {1, 0, 1, 0, 1 },
                    //            {1, 0, 1, 0, 1 },
                    //            {1, 2, 1, 1, 1 },
                    //            {1, 1, 1, 1, 1 },
            },
    }) {}
}