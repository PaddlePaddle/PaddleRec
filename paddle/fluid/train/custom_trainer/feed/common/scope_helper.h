#pragma once
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {
    
class ScopeHelper {
public:
    //直接取var
    template <class T>
    static const T& var(paddle::framework::Scope* scope, const std::string& name) {
        return scope->Var(name)->Get<T>();
    }
    template <class T>
    static T* mutable_var(paddle::framework::Scope* scope, const std::string& name) {
        return scope->Var(name)->GetMutable<T>();
    }

    template <class T>
    static T* resize_variable(paddle::framework::Scope* scope,
        const std::string& name, const paddle::framework::DDim& dim) {
        auto* tensor = scope->Var(name)->GetMutable<T>();
        tensor->Resize(dim);
        return tensor; 
    }
    
    static paddle::framework::LoDTensor* resize_lod_tensor(
        paddle::framework::Scope* scope,
        const std::string& name, const paddle::framework::DDim& dim) {
        return resize_variable<paddle::framework::LoDTensor>(scope, name, dim);
    }

    template <class T>
    static void fill_value(paddle::framework::Scope* scope,
        paddle::platform::Place place, const std::string& name, T& value) {
        auto* tensor = resize_variable<paddle::framework::Tensor>(scope, name, { 1 });
        T* data = tensor->mutable_data<T>(place);
        *data = value;
        return;
    } 
    
    template <class T>
    static T* get_value(paddle::framework::Scope* scope,
        paddle::platform::Place place, const std::string& name) {
        auto* tensor = scope->Var(name)->GetMutable<paddle::framework::Tensor>();
        return tensor->mutable_data<T>(place);
    }

};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
