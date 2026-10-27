"""Optional browser checks: install playwright and its Chromium to run these."""

import os

import pytest
from symbolica import E
from symbolica.community.tensor import TensorExpression

pw = pytest.importorskip("playwright.sync_api")


def test_widget_replaces_pages_rejects_stale_replies_and_disposes():
    expression = TensorExpression(
        E("+".join(f"frontend_test::x^{i}" for i in range(201)))
    )
    original = expression.to_expression()
    viewer = expression.paged(page_size=25)
    fallback = viewer._repr_mimebundle_(include=["text/html"])["text/html"]
    assert "Navigation requires a live notebook" in fallback

    with pw.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(
                **(
                    {"executable_path": os.environ["CHROMIUM_EXECUTABLE"]}
                    if "CHROMIUM_EXECUTABLE" in os.environ
                    else {}
                )
            )
        except pw.Error as error:
            pytest.skip(f"Chromium unavailable: {error}")
        page = browser.new_page(viewport={"width": 1200, "height": 1000})
        page.set_default_timeout(10_000)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("https://cdn.jsdelivr.net/**", lambda route: route.abort())
        page.set_content('<div id="viewer" style="width:900px"></div>')
        page.evaluate(
            """async ({esm, initial}) => {
            const url = URL.createObjectURL(new Blob([esm], {type:'text/javascript'}));
            const module = await import(url);
            URL.revokeObjectURL(url);
            const state = {page:initial, preview:{}};
            const callbacks = new Map();
            window.messages = [];
            window.setResponse = ([field, next]) => {
                if (!field) return;
                state[field] = next;
                for (const callback of callbacks.get(`change:${field}`) || []) callback();
            };
            window.callbackCount = () => [...callbacks.values()].reduce((n, s)=>n+s.size, 0);
            window.dispose = module.default.render({el:document.querySelector('#viewer'), model:{
                get:name=>state[name], send:message=>messages.push(message),
                on:(name,callback)=>{
                    if (!callbacks.has(name)) callbacks.set(name, new Set());
                    callbacks.get(name).add(callback);
                },
                off:(name,callback)=>callbacks.get(name)?.delete(callback)
            }});
        }""",
            {"esm": viewer.javascript(), "initial": viewer.page()},
        )

        def reply_to_latest(action):
            message = page.evaluate("messages.at(-1)")
            assert message["action"] == action
            response = viewer.handle_message(message)
            assert response is not None and "error" not in response[1], response
            page.evaluate("response=>setResponse(response)", response)
            return response

        try:
            reply_to_latest("attach")
            assert page.evaluate("callbackCount()") == 2
            assert page.locator(".toolbar").is_hidden()

            # Root expressions expose omitted parts through a preview. Preview
            # requests carry IDs; ordinary page messages no longer need them.
            page.locator(".math-content button.omission").click()
            first_preview = reply_to_latest("preview")
            popup = page.locator(".preview")
            assert popup.get_attribute("aria-busy") is None
            popup_next = page.locator(".preview-next")
            popup_next.click()
            assert popup_next.is_disabled()
            before = page.locator(".preview-math").inner_html()
            stale = [
                first_preview[0],
                {**first_preview[1], "html": "<math><mi>stale</mi></math>"},
            ]
            page.evaluate("response=>setResponse(response)", stale)
            assert popup.get_attribute("aria-busy") == "true"
            assert page.locator(".preview-math").inner_html() == before
            current_preview = reply_to_latest("preview")
            assert current_preview[1]["start"] == 25
            after = page.locator(".preview-math").inner_html()
            assert after != before
            page.evaluate("response=>setResponse(response)", stale)
            assert page.locator(".preview-math").inner_html() == after

            # Inspect uses the actual provider's omission capability to open a
            # separately paged expression, with main Next/Previous controls.
            page.get_by_role("button", name="Inspect", exact=True).click()
            reply_to_latest("open")
            next_button = page.locator(".toolbar").get_by_role(
                "button", name="Next", exact=True
            )
            states = [viewer.page()]
            for _ in range(4):
                before = page.locator(".math-content").inner_html()
                next_button.click()
                _, state = reply_to_latest("next")
                assert state["start"] == states[-1]["end"]
                states.append(state)
                assert page.locator(".math-content").inner_html() != before
                # Continuation and omission markers become separate flow items.
                assert (
                    0
                    < page.locator(".math-content math").count()
                    <= (state["end"] - state["start"] + 3)
                )
                assert page.locator(".math-content math *").count() < 1000
            page.locator(".toolbar").get_by_role(
                "button", name="Previous", exact=True
            ).click()
            _, previous = reply_to_latest("previous")
            assert previous["html"] == states[-2]["html"]

            # The shared viewer fits content to width and provides overflow on
            # its outer container, replacing the old horizontal-scroll toggle.
            assert (
                page.locator(".math").evaluate("e=>getComputedStyle(e).maxHeight")
                == "none"
            )
            assert (
                page.locator(".math").evaluate("e=>getComputedStyle(e).overflowX")
                == "auto"
            )
            assert page.locator(".math-content math").evaluate_all(
                "elements=>elements.every(e=>getComputedStyle(e).overflowX==='visible')"
            )
            before = page.locator(".math-content").text_content()
            page.locator("#viewer").evaluate("element=>element.style.width='480px'")
            page.evaluate(
                "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
            )
            assert page.locator(".math-content").text_content() == before
            assert page.locator(".math").evaluate(
                "e=>e.scrollWidth <= e.clientWidth + 2"
            )
            wide = TensorExpression(E("scroll_test::" + "W" * 200)).paged()
            try:
                page.evaluate("response=>setResponse(response)", ["page", wide.page()])
                assert page.locator(".math").evaluate(
                    "e=>e.scrollWidth > e.clientWidth"
                )
                assert page.locator(".math").evaluate(
                    "e=>{e.scrollLeft=50;return e.scrollLeft>0;}"
                )
            finally:
                wide.close()
            page.evaluate("response=>setResponse(response)", ["page", previous])
            assert page.locator(".math").evaluate("e=>e.scrollLeft") == 0
            assert expression.to_expression() == original

            page.evaluate("dispose()")
            assert page.evaluate("callbackCount()") == 0
            disposal = page.evaluate("messages.at(-1)")
            assert disposal["action"] == "dispose"
            assert viewer.handle_message(disposal)[0] == ""
            assert page.locator("math").count() == 0
            page.evaluate("response=>setResponse(response)", current_preview)
            assert page.locator("math").count() == 0
            assert not errors, errors
        finally:
            viewer.close()
            browser.close()
