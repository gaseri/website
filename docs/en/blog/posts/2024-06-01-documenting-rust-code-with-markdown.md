---
author: Kristijan Lenković
authors:
  - klenkovic
date: 2024-06-01
readtime: 15
tags:
  - rust
  - markdown
  - documentation
  - rustdoc
  - axum
  - web server
description: >-
  A small demonstration of the usage of Markdown while documenting simple Rust
  web service powered by axum and generating HTML documentation with rustdoc tool.
---

# Documenting Rust code with Markdown

![Happy Rustacean](https://www.rustacean.net/assets/rustacean-flat-happy.svg){ align=left style="width: 190px" }

Recently, I decided to move on from [Node.js](https://nodejs.org/) and start learning something new related to web development. Over the past several years, [Rust](https://www.rust-lang.org/) has gained significant popularity, to the point where [backend web services](https://github.com/tokio-rs/axum/blob/main/ECOSYSTEM.md#project-showcase) and numerous [other extraordinary CLI tools and apps](https://github.com/rust-unofficial/awesome-rust) are being built with it.

As I started learning and tinkering with Rust, I discovered a feature I wasn't aware of - the handy and powerful documentation tool called [rustdoc](https://doc.rust-lang.org/rustdoc/what-is-rustdoc.html), which comes with the standard Rust distribution. What's cool about `rustdoc` is that it uses Markdown syntax, allowing developers to document their code with Markdown and generate HTML documentation. Think of it as "*MkDocs meets Doxygen.*"

Since the [esteemed members](../../people/index.md) of the **😎 GASERI** group [love their Markdown](../../../hr/aktivizam/mapapijri.md), I thought it would be nice to share this lovely "discovery" with the rest of the community by demonstrating how to document a "Hello World" Rust web service.

<!-- more -->

## Installation

First, we must ensure that Rust is installed on our machine. The recommended way is to use [rustup](https://rust-lang.github.io/rustup/), the Rust toolchain installer. As mentioned, `rustdoc` is part of the standard Rust distribution and will be installed with `rustup`, along with the [rustc](https://doc.rust-lang.org/rustc/what-is-rustc.html) compiler and [cargo](https://doc.rust-lang.org/cargo/) package manager. You can get `rustup` on [the Rust website here](https://www.rust-lang.org/tools/install) or follow [the official guide for other installation methods here](https://forge.rust-lang.org/infra/other-installation-methods.html).

[Install :fontawesome-brands-rust:](https://www.rust-lang.org/tools/install){ .md-button .md-button--primary }

## Project Setup

Once you have installed Rust on your system, let's create a new project and install some dependencies that we will need for this to work.

``` shell
cargo new gaseri-rustdoc-demo
cd gaseri-rustdoc-demo
cargo add axum@0.7.5 tokio@1.37.0 -F tokio@1.37.0/macros -F tokio@1.37.0/rt-multi-thread
```

Make sure your `Cargo.toml` looks like this before we proceed:

``` toml title="Cargo.toml"
[package]
name = "gaseri-rustdoc-demo"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = "0.7.5"
tokio = { version = "1.37.0", features = ["macros", "rt-multi-thread"] }
```

!!! warning
    In the broad world of web development, Rust is still a rookie compared to PHP, Python, and even Node.js. Libraries such as [Axum](https://docs.rs/axum/latest/axum/) are still in active development, below [SemVer](https://semver.org/) 1.0.0, and are prone to frequent changes.

    The following code most likely won't work unless you have the exact versions of the required dependencies, as specified above.

## Coding, Compiling, and Running

Now, let's build our simple web service. Open the `src/main.rs` file and overwrite it with the following code:

``` rust title="main.rs" linenums="1"
//! GASERI rustdoc demonstration web service
//! 
//! This simple web service made with the
//! [Rust](https://www.rust-lang.org/) language,
//! powered by [Axum](https://docs.rs/axum/latest/axum/),
//! is created for the sole purpose of demonstrating the 
//! capability of `rustdoc` tool to the 
//! [GASERI](https://gaseri.org) community.
//! 
//! # Usage
//! 
//! First, you need to compile and run this program with `cargo`
//! to make sure everything works as expected:
//! 
//! ```
//! cargo run
//! ```
//! Once you veried the code compiles and runs properly by opening
//! [http://localhost:3000](http://localhost:3000), you can generate
//! the documentation:
//! 
//! ```
//! cargo doc
//! ```
//! 
//! When complete, the generator will provide the path to your documentation.
//! Open this URL in you browser:
//! 
//! ```
//! file://path/to/gaseri-rustdoc-demo/target/doc/gaseri_rustdoc_demo/index.html
//! ```

use axum::{response::Redirect, routing::get, Router};

/// Handler/resolver that returns simple hello message
pub async fn hello_gaseri() -> String {
    "Hello GASERI! 😎".to_owned()
}

/// Handler/resolver that redirects the browser to the GASERI.org website
/// 
/// Permanent (301) redirect is used.
pub async fn redirect_gaseri() -> Redirect {
    Redirect::permanent("https://gaseri.org/")
}

/// Handler/resolver that is still not implemented
/// 
/// <div class="warning">Warning: Opening this path in the browser will cause a panic!</div>
pub async fn tbd() -> String {
    todo!()
}

/// Creates a Router with custom Routes
/// 
/// Make sure you add/append layers at the bottom to apply them to upper routes.
/// Currently, no layers/middlwares are attached.
pub fn create_routes() -> Router {
    Router::new()
    .route("/", get(hello_gaseri))
    .route("/gaseri", get(redirect_gaseri))
    .route("/tbd", get(tbd))
}

/// Creates an app by creating a Router and a listener
/// 
/// Use it to run your server
/// 
/// # Example
/// ```
/// use hello_world::run;
/// 
/// #[tokio::main]
/// async fn main() {
///     run().await
/// }
/// ```
pub async fn run() {
    // Create the app with multiple routes
    let app = create_routes();

    // Run the app and listen on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// The **main** function of this program
/// 
/// Runs the web service
#[tokio::main]
async fn main() {
    run().await
}
```

Let's make sure everything compiles and works as expected before we continue.

``` shell
cargo run
```

Open your favorite browser and navigate to [http://localhost:3000](http://localhost:3000). It should display the text `Hello GASERI! 😎`.

I won't describe how Rust or [Axum](https://docs.rs/axum/latest/axum/) work since that is beyond the scope of this blog post. However, you can read the comments in the code or the documentation we are about to generate, and it should be pretty clear how the web service is working. Isn't that the whole point of code documentation?

## Generating the Code Documentation

Now, let's get to the fun part! Run the following command:

``` shell
cargo doc
```

Congratulations, you just generated the HTML code documentation, which you can use (or even host)! Go ahead and open it with your favorite browser.

The generated HTML documentation includes not only your own documentation but the **entire collection of documented crates** we are using as dependencies for our service!

*Isn't that cool*?! 😎

## Learn more

To learn more about writing a documentation in Rust and using `rustdoc`, read the [official rustdoc book](https://doc.rust-lang.org/rustdoc/what-is-rustdoc.html).

[More on doc writing](https://doc.rust-lang.org/rustdoc/how-to-write-documentation.html){ .md-button .md-button--primary }
[Read the rustdoc book](https://doc.rust-lang.org/rustdoc/what-is-rustdoc.html){ .md-button }

## Conclusion

I was *astonished* when I discovered this! Hopefully, if you've made it this far through this blog post, you are too. Coming from the Node.js world, this is spot on what I had always been missing in that ecosystem. Now, before any hardcore Node.js fans intervene: *YES, I am aware of the existence* of [JSDoc](https://jsdoc.app/), [ESDoc](https://esdoc.org/), [documentation.js](https://documentation.js.org/), and similar tools that do exactly the same thing; I have used them before. The big difference is that the Rust ecosystem was built with this tool in mind from day one. I'm sure there will be similar alternatives or better tools in the future, but the fact that `rustdoc` is a native part of the standard Rust distribution, along with `rustc` and `cargo`, like `npm` is for Node.js, is precisely why I like this.

As you may have noticed, `rustdoc` generated the code documentation for the entire project and its dependencies. This is extremely important because those dependencies change over time, especially in the context of Rust web development. Instead of googling and searching for the right dependency version, you can generate the entire documentation by yourself and have a precise version of the library you are depending on, completely offline, while you enjoy your vacation in the middle of the Mediterranean Sea (if you like that sort of thing as I do). And the best of all, it natively supports Markdown and converts it to HTML! 😎
