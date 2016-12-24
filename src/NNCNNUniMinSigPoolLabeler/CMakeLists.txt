
include_directories(
  model
)

add_executable(NNCNNUniMinSigPoolLabeler NNCNNUniMinSigPoolLabeler.cpp)
target_link_libraries(NNCNNUniMinSigPoolLabeler ${LIBS})
