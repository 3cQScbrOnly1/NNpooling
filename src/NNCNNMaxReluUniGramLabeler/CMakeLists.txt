
include_directories(
  model
)

add_executable(NNCNNMaxReluUniGramLabeler NNCNNMaxReluUniGramLabeler.cpp)
target_link_libraries(NNCNNMaxReluUniGramLabeler ${LIBS})
