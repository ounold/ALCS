#include <vector>

#include <Maze/Mazes/Woods1.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"

namespace Environments {
    using enum EFieldState;

    Woods1::Woods1() : IMazeEnvironment(
    7 * 4,
    {
            .mazeWidth = 7,
            .mazeHeight = 7,
            .goalState = {3, 3},
            .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, PRIZE,    CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    //            { 1, 1, 1, 1, 1, 1, 1 },
                    //            { 1, 0, 0, 0, 0, 0, 1 },
                    //            { 1, 0, 0, 0, 0, 0, 1 },
                    //            { 1, 1, 1, 2, 0, 0, 1 },
                    //            { 1, 1, 1, 1, 0, 0, 1 },
                    //            { 1, 1, 1, 1, 0, 0, 1 },
                    //            { 1, 1, 1, 1, 1, 1, 1 }
            },

    }) {}
}