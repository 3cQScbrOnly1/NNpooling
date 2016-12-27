
include_directories(
  model
)

add_executable(NNCNNMaxReluUniBiTriGramLabeler NNCNNMaxReluUniBiTriGramLabeler.cpp)
target_link_libraries(NNCNNMaxReluUniBiTriGramLabeler ${LIBS})
