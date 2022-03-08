using Plots

N = 100 
# α = 1.0
a = 0
b = 2pi
Nrange = 5:100
∞ = 555676575858 # arbitrary
funct(x) = sin(x)
deriv(x) = cos(x)
u = 1 # temporarily set for velocity, will make it a function R2 for flow solver

# these are defaults
# we can call the functions with other values


function tay_exp_approx(f::Vector,dx::Real)

    n = length(f) # 
    
    u=[]
    i = 1
    ui=(f[i+1]-f[i])/dx

    append!(u,ui)
    for i ∈ 2:n-1
        ui = (f[i+1]-f[i-1])/(2dx)
        append!(u,ui)
    end

    i=n
    ui=(f[i]-f[i-1])/dx
    append!(u,ui)
    
    
    return u

end

using Calculus, LinearAlgebra
function compute_error(func,u,x,p,∞)
    deriv(x) = cos(x) # should be derivative of func
    u_analytic = []
    for i ∈ x
        item = deriv(i)
        append!(u_analytic,item)
    end
    
    
    if p != ∞
        
        lp = norm(u-u_analytic,p)
        return lp
    else
        l_inf = maximum(abs.(u-u_analytic))
        return l_inf
    end
end

function get_norm(dx,f,func,x,p,∞)
    
    deriv = tay_exp_approx(f,dx)
    error = compute_error(func,deriv,x,p,∞)

    return error
    
end

# collect error function 
# p is the desired norm to call with the main functionv
function collect_error(N,a,b,p)
    

    dx = (b-a)/N;
    x = []

    for i ∈ 1:N
        item = (i-1)*(dx)+a
        append!(x,item)
    end
    func(i) = sin(i)
    f = []
    
    for i ∈ x
        item = func(i) # differential to be approximated
        append!(f,item)
    end
    
    u = tay_exp_approx(f,dx)
    
    ∞ = 555676575858
    
    return get_norm(dx,f,func,x,p,∞)
 
end

function get_numerical_deriv(N,a,b,func)
    

    dx = (b-a)/N;
    x = []

    for i ∈ 1:N
        item = (i-1)*(dx)+a
        append!(x,item)
    end
    f = []
    for i ∈ x
        item = func(i) # differential to be approximated
        append!(f,item)
    end
    
    u = tay_exp_approx(f,dx)
    return u
end


function plot_error_norms(a,b,Nrange)
    
    p1norms = []
    p2norms = []
    p3norms = []
    p10norms = []
    pinfnorms = []
    for N ∈ Nrange # loop that runs main function over many N
        p1norm = collect_error(N,a,b,1) # main(N, start, stop, desired norm)
        p2norm = collect_error(N,a,b,2)
        p3norm = collect_error(N,a,b,3)
        p10norm = collect_error(N,a,b,10)
        pinfnorm = collect_error(N,a,b,∞)
        append!(p1norms,p1norm)
        append!(p2norms,p2norm)
        append!(p3norms,p3norm)
        append!(p10norms,p10norm)
        append!(pinfnorms,pinfnorm)
    end # there is certianly a more efficient way to call our function over a range of points
    # next time, can rewrrite these calls using broadcast()


    plot(Nrange,[p1norms, p2norms, p3norms, p10norms, pinfnorms],
        labels = ["l1norm" "l2norm" "l3norm" "l10norm" "pinfnorm"], 
        ylabel = "error", xlabel = "Nsteps"#=scale=:log10, title=:"Log-Log: Error vs N"=#)
    plot!()
    return p2norms
end


function plot_num_deriv(N,a,b,func,deriv)
    range = LinRange(a,b,N)
    plot(range,get_numerical_deriv(N,a,b,func),labels = "numerical solution")
    plot!(deriv,labels = "analytical solution")
end

function get_slope(error_norm,Nrange)
    num = log(error_norm[length(error_norm)])-log(error_norm[1])
    den = log(Nrange[length(Nrange)])-log(Nrange[1])
    return -1/(num/den)
end

function vectorize_function(a,b,N,func)
    dx = (b-a)/N;
    #x = Array{Float64,1}(undef,length(N))
    x = []
    for i ∈ 1:N
        item = (i-1)*(dx)+a
        append!(x,item)
    end
    f = []
    for i ∈ x
        item = func(i) # differential to be approximated
        append!(f,item)
    end

    return f
end


function explicit_solve(t0,tf,tsteps,u,func)
    
    f = vectorize_function(0,2pi,100,sin) # can put in custom calls 
    dx = 2pi/100
    dt = (tf-t0)/tsteps
    t = Array{Float64,1}(undef,tsteps)
    
    t[1] = t0
    phi = Array{Vector,1}(undef,tsteps) # big discovery
    phi[1] = f
    
    for i ∈ 1:tsteps-1
        t[i+1] = t[i] + dt;
        #ft = -u*tay_exp_approx(phi[i],dx)
        ft = -u*cyclic_deriv(phi[i],dx)
        phi[i+1] = phi[i] + dt*ft
    end 
    return phi
end


function plot_phi(ϕ)
    anim = @animate for i in 1:length(ϕ)
        plot(ϕ[i])
    end

    gif(anim,"anim_fps15.gif", fps = 15)
end

plot_phi(ϕ)


function cyclic_deriv(f::Vector,dx::Float64)
    n = length(f) # 
    u=[]
    i = 1
    ui=(f[i+1]-f[n])/(2*dx)
    append!(u,ui)
    for i ∈ 2:n-1
        ui = (f[i+1]-f[i-1])/(2dx)
        append!(u,ui)
    end

    i=n
    ui=(f[1]-f[i-1])/(2dx)
    append!(u,ui)
    
    return u
end

function rkexplicit(t0,tf,tsteps,u,func,nsteps,a,b)
    
    f = vectorize_function(a,b,nsteps,func) # can put in custom calls 
    dx = (b-a)/nsteps
    dt = (tf-t0)/tsteps
    t = Array{Float64,1}(undef,tsteps)
    #phi = Array{Float64,1}(undef,nsteps)
    #phi[1,1] = t0
    phi = f

    for i ∈ 2:tsteps
        #t = phi[i-1,1]
        w = phi[i-1]

        k1 = dt*(-u)*cyclic_deriv(w,dx)
        k2 = dt*(-u)*cyclic_deriv(w+k1/2,dx)
        k3 = dt*(-u)*cyclic_deriv(w+k2/2,dx)
        k4 = dt*(-u)*cyclic_deriv(w+k3/2,dx)

        phi[i,1] = t0 + (i-1)*dt
        phi[i,2] = w + (k1+2k2+2k3+k4)/6
        

    end 
    return phi
end



function plot_rkphi(ϕ)
    anim = @animate for i in 1:size(ϕ)[1]
        plot(ϕ)
    end
    gif(anim,"anim_fps15.gif", fps = 15)
end

plot_rkphi(ϕ)

function vectorize_3dfunction(a,b,N,func)
    dx = (b-a)/N;
    x = []

    for i ∈ 1:N
        item = (i-1)*(dx)+a
        append!(x,item)
    end
    f = Array{Float64,2}(undef,length(x),2)
    for i ∈ x
        f[i,1] = func(i,i)
        f[i,2] = func(i,i) 
        
    end

    return f
end

t(x) = sin(x)
ϕ = rkexplicit(0,10,100,1,sin,400,0,2pi)
z(x,y) = sin(sqrt(x^2+y^2))
test = rkexplicit(0,20,100,1,z,100,0,2pi)
gauss(x) = exp(-10(x-pi)^2)


function rkexplicittest(t0,tf,tsteps,u,func,nsteps,a,b)
    # define grid 
    dx = (b-a)/nsteps
    f = map(func,LinRange(a,b,nsteps+1))
    pop!(f) # corrects for the errror between vectorize_function and map
    #f = vectorize_function(a,b,nsteps,func) # can put in custom calls 
   
    dt = (tf-t0)/tsteps
    #phi = Array{Float64,1}(undef,nsteps)
    phi = f

    anim = @animate for i ∈ 2:tsteps
        #t = phi[i-1,1]
        w = phi

        k1 = dt*(-u)*cyclic_deriv(w,dx)
        k2 = dt*(-u)*cyclic_deriv(w+k1/2,dx)
        k3 = dt*(-u)*cyclic_deriv(w+k2/2,dx)
        k4 = dt*(-u)*cyclic_deriv(w+k3/2,dx)

        phi = w + (k1+2k2+2k3+k4)/6
        plot(phi,legend = false)
    end 
    gif(anim,"anim_fps15.gif", fps = 15)
end

rkexplicittest(0,5,100,1,gauss,100,0,2pi)

using BenchmarkTools
@btime rkexplicittest(0,10,100,1,sin,100,0,2pi);
@btime rkexplicit(0,10,100,1,sin,100,0,2pi);

y = vectorize_function(0,2pi,10,sin)
x = (map(sin,LinRange(0,2pi,10)))
pop!(x)
x
plot(x)

plot(x)
plot!(y)